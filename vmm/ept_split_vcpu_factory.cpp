#include <bfcallonce.h>
#include <bfbitmanip.h>

#include <bfvmm/Vcpu/vcpu_factory.h>
#include <bfvmm/memory_manager/arch/x64/unique_map.h>

#include <eapis/hve/arch/intel_x64/vcpu.h>
using namespace eapis::intel_x64;

#include <array>
#include <mutex>
#include <vector>

// Macros for defining print levels.
//
#define VIOLATION_EXIT_LVL  0
#define SPLIT_CONTEXT_LVL   1
#define MONITOR_TRAP_LVL    1
#define THRASHING_LVL       0
#define WRITE_LVL           1
#define DEBUG_LVL           0
#define ALERT_LVL           0
#define ERROR_LVL           0

// -----------------------------------------------------------------------------
// vCPU
// -----------------------------------------------------------------------------

namespace ept_split
{

// This flag is used in the call_once call.
//
bfn::once_flag flag{};

// We one main memory map and one copy,
// which is for single stepping, for the
// case we detect thrashing.
//
ept::mmap g_mainMap{};
ept::mmap g_trapMap{};
ept::mmap::entry_type g_dummyPte{};

// Mutex for EPT modifications.
//
std::mutex g_eptMutex;

// Maximum number of splits you can have
// at one time.
//
constexpr unsigned const MAX_SPLITS = 100u;

class SplitPool {
public:
    SplitPool() noexcept : m_splitContexts{}, m_refMutex{}
    { }

    struct SplitContext {
        bool enabled{};
        uintptr_t cleanPhys{};

        std::reference_wrapper<ept::mmap::entry_type> pte{g_dummyPte};
        uintptr_t shadowPhys{};
        uintptr_t shadowVirt{};
        uintptr_t cleanVirt{};

        size_t refCount{};
        uint64_t cr3{};
        std::unique_ptr<uint8_t[]> shadowPage{nullptr};
    };

    SplitContext* getContext(const uintptr_t gpa4k) noexcept {
        for (auto it = m_splitContexts.begin();
            it != m_splitContexts.end(); ++it) {
            if (it->enabled && (it->cleanPhys == gpa4k)) return it;
        }

        return nullptr;
    }

    SplitContext* getFreeContext() noexcept {
        for (auto it = m_splitContexts.begin();
            it != m_splitContexts.end(); ++it) {
            if (!it->enabled) {
                it->cleanPhys = 0ull;
                it->enabled = true;
                return it;
            }
        }

        return nullptr;
    }

    SplitContext* getFirstEnabledContext() noexcept {
        for (auto it = m_splitContexts.begin();
            it != m_splitContexts.end(); ++it) {
            if (it->enabled)
                return it;
        }

        return nullptr;
    }

    SplitContext* getContextByIndex(const int index) noexcept {
        return m_splitContexts.begin() + index;
    }

    // Counts how many split contexts are currently
    // enabled/active.
    //
    unsigned activeSplits() const noexcept {
        unsigned counter = 0;
        for (const auto& ctx : m_splitContexts) {
            if (ctx.enabled) counter++;
        }
        return counter;
    }

    // Member functions for modifying the reference counter of split contexts.
    //
    size_t incCounter(SplitContext* ctx) noexcept {
        std::lock_guard lock{m_refMutex};
        return ++ctx->refCount;
    }

    size_t decCounter(SplitContext* ctx) noexcept {
        std::lock_guard lock{m_refMutex};
        return --ctx->refCount;
    }

    size_t resetCounter(SplitContext* ctx) noexcept {
        std::lock_guard lock{m_refMutex};
        return ctx->refCount = 1;
    }

private:
    std::array<SplitContext, MAX_SPLITS> m_splitContexts;
    std::mutex m_refMutex;
};

// This pool will hold all split contexts.
//
SplitPool g_splits;

// Mask for writing to PTEs (4k).
//
constexpr const auto pteMask4k = ::intel_x64::ept::pt::entry::phys_addr::mask
    | ::intel_x64::ept::pt::entry::read_access::mask
    | ::intel_x64::ept::pt::entry::write_access::mask
    | ::intel_x64::ept::pt::entry::execute_access::mask;

// Mask for writing to PTEs (4k).
//
constexpr const auto pteMask2m = ::intel_x64::ept::pd::entry::phys_addr::mask
    | ::intel_x64::ept::pd::entry::read_access::mask
    | ::intel_x64::ept::pd::entry::write_access::mask
    | ::intel_x64::ept::pd::entry::execute_access::mask;

// Helper enum for page selection.
//
enum PageT
{
    clean,
    shadow
};

// Helper enum for access bit selection.
//
enum AccessBitsT
{
    read        = 0b001,
    write       = 0b010,
    read_write  = 0b011,
    execute     = 0b100,
    all         = 0b111
};

class Vcpu : public eapis::intel_x64::vcpu
{

    // Helper function for writing to page table entries (PTE).
    //
    void writePte(
        uintptr_t& pte, const uintptr_t physAddr, const AccessBitsT bits, const uint64_t mask = pteMask4k) noexcept
    { pte = set_bits(pte, mask, physAddr | bits); }

    // Helper function for flipping between pages
    //
    // Sets the physical address and access bits for
    // the provided split context.
    //
    void flipPage(SplitPool::SplitContext* ctx, const PageT flipTo) noexcept {
        if (flipTo == PageT::clean)
        { writePte(ctx->pte.get(), ctx->cleanPhys, AccessBitsT::read_write); }
        else
        { writePte(ctx->pte.get(), ctx->shadowPhys, AccessBitsT::execute); }
    }

    void flipPage(uintptr_t gpa4k, const PageT flipTo) noexcept {
        flipPage(g_splits.getContext(gpa4k), flipTo);
    }

    // For thrashing detection.
    //
    uintptr_t m_prevRip{};
    size_t m_ripCounter{};

    // For resetting the access bits
    // after setting the monitor trap.
    //
    SplitPool::SplitContext* m_trapCtx{nullptr};

public:

    // Constructor
    //
    // This is the only constructor the vCPU supports, so it must be
    // overloaded.
    //
    Vcpu(vcpuid::type id) :
        eapis::intel_x64::vcpu{id}
    {
        using namespace vmcs_n;
        using mt_delegate_t = monitor_trap_handler::handler_delegate_t;
        using eptv_delegate_t = ept_violation_handler::handler_delegate_t;

        // Add VMCall handler.
        exit_handler()->add_handler(
            exit_reason::basic_exit_reason::vmcall,
            ::handler_delegate_t::create<Vcpu, &Vcpu::vmcall_handler>(this)
        );

        // Add Monitor Trap handler.
        //
        eapis()->add_monitor_trap_handler(
            mt_delegate_t::create<Vcpu, &Vcpu::monitor_trap_handler>(this)
        );

        // Add EPT violation handlers.
        //
        eapis()->add_ept_execute_violation_handler(
            eptv_delegate_t::create<Vcpu, &Vcpu::ept_execute_violation_handler>(this)
        );

        eapis()->add_ept_read_violation_handler(
            eptv_delegate_t::create<Vcpu, &Vcpu::ept_read_violation_handler>(this)
        );

        eapis()->add_ept_write_violation_handler(
            eptv_delegate_t::create<Vcpu, &Vcpu::ept_write_violation_handler>(this)
        );

        // Setup the EPT memory map once.
        //
        bfn::call_once(flag, [&] {
            ept::identity_map(g_mainMap, MAX_PHYS_ADDR);
            ept::identity_map(g_trapMap, MAX_PHYS_ADDR);
        });

        eapis()->enable_vpid();
        eapis()->set_eptp(g_mainMap);
    }

    // -----------------------------------------------------------------------------
    // VMCall Handler
    // -----------------------------------------------------------------------------

    bool
    vmcall_handler(
        gsl::not_null<vmcs_t *> vmcs)
    {
        // Register usage:
        // - rax: Legacy vmcall mode
        // - rdx: Legacy magic number
        // - rcx: Command ID switch
        // - rbx: arg1
        // - rsi: arg2
        //
        // RCX will also be the return value register.
        //
        guard_exceptions([&] {
            auto* state = vmcs->save_state();
            switch(vmcs->save_state()->rcx) {
                // hv_present()
                case 0ull:
                {
                    state->rcx = hvPresent();
                    break;
                }

                // create_split_context(int_t gva)
                case 1ull:
                {
                    state->rcx = createSplitContext(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // activate_split(int_t gva)
                case 2ull:
                {
                    state->rcx = activateSplit();
                    break;
                }

                // deactivate_split(int_t gva)
                case 3ull:
                {
                    state->rcx = deactivateSplit(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // deactivate_all_splits()
                case 4ull:
                {
                    state->rcx = deactivateAllSplits();
                    break;
                }

                // is_split(int_t gva)
                case 5ull:
                {
                    state->rcx = isSplit(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // write_to_shadow_page(int_t from_va, int_t to_va, size_t size)
                case 6ull:
                {
                    state->rcx = writeMemory(state->rbx, state->rsi, state->r08, vmcs_n::guest_cr3::get());
                    break;
                }

                // TODO: Implement functions for below switches.

                // get_flip_num()
                case 7ull:
                    state->rcx = -1ull;
                    break;

                // get_flip_data(int_t out_addr, int_t out_size)
                case 8ull:
                    state->rcx = -1ull;
                    break;

                // clear_flip_data()
                case 9ull:
                    state->rcx = -1ull;
                    break;

                // remove_flip_entry(int_t rip)
                case 10ull:
                    state->rcx = -1ull;
                    break;

                default:
                    state->rcx = -1ull;
                    break;
            };
        });

        // Make sure we advance the instruction pointer. Otherwise, the VMCall
        // instruction will be executed in an infinite look. Also note that
        // the advance() function always returns true, which tells the base
        // hypervisor that this VM exit was successfully handled.
        //
        return advance(vmcs);
    }

    // -----------------------------------------------------------------------------
    // EPT Violation Handlers
    // -----------------------------------------------------------------------------

    bool ept_read_violation_handler(gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info) {
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "read: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa4k", gpa4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (auto* ctx = g_splits.getContext(gpa4k); ctx != nullptr)
        {
            // Check for thrashing.
            //
            if (vmcs->save_state()->rip == m_prevRip && ctx->cr3 == cr3) { m_ripCounter++; }
            else { m_prevRip = vmcs->save_state()->rip; m_ripCounter = 1; }

            if (m_ripCounter > 4)
            {
                bfalert_nhex(THRASHING_LVL, "read: thrashing detected", m_prevRip);
                m_ripCounter = 1;

                // Enable monitor trap flag for single stepping.
                //
                // m_trapCtx = ctx;
                // writePte(ctx->pte.get(), ctx->cleanPhys, AccessBitsT::all);
                eapis()->set_eptp(g_trapMap);
                eapis()->enable_monitor_trap_flag();
            }
            else
            {
                // Flip to clean page.
                //
                std::lock_guard lock{g_eptMutex};
                flipPage(ctx, PageT::clean);

                if (::intel_x64::ept::pt::entry::execute_access::is_enabled(ctx->pte))
                    bferror_nhex(0, "read: exec not disabled", gpa4k);
                if (::intel_x64::ept::pt::entry::read_access::is_disabled(ctx->pte))
                    bferror_nhex(0, "read: read not enabled", gpa4k);
                if (::intel_x64::ept::pt::entry::write_access::is_disabled(ctx->pte))
                    bferror_nhex(0, "read: write not enabled", gpa4k);
            }
        } else {
            bfalert_nhex(ALERT_LVL, "read: no split context found", gpa4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_mainMap.entry(gpa4k);

            // Check page granularity.
            //
            if (g_mainMap.is_4k(gpa4k)) {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!read_access::is_enabled(entry))
                {
                    bfalert_nhex(ALERT_LVL, "read: resetting access bits for 4k entry", gpa4k);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            } else {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!read_access::is_enabled(entry))
                {
                    const auto gpa2m = bfn::upper(gpa4k, ::intel_x64::ept::pt::from);
                    bfalert_nhex(ALERT_LVL, "read: resetting access bits for 2m entry", gpa2m);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            }
        }

        info.ignore_advance = true;
        return true;
    }

    bool ept_write_violation_handler(gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info) {
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "write: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa4k", gpa4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (auto* ctx = g_splits.getContext(gpa4k); ctx != nullptr)
        {
            // Check if the write violation occurred in the same CR3 context.
            //
            if (ctx->cr3 == cr3) {
                // Check for thrashing.
                //
                if (vmcs->save_state()->rip == m_prevRip) { m_ripCounter++; }
                else { m_prevRip = vmcs->save_state()->rip; m_ripCounter = 1; }

                if (m_ripCounter > 4)
                {
                    bfalert_nhex(THRASHING_LVL, "write: thrashing detected", m_prevRip);
                    m_ripCounter = 1;

                    // Enable monitor trap flag for single stepping.
                    //
                    // m_trapCtx = ctx;
                    // writePte(ctx->pte.get(), ctx->cleanPhys, AccessBitsT::all);
                    eapis()->set_eptp(g_trapMap);
                    eapis()->enable_monitor_trap_flag();
                }
                else
                {
                    // Flip to clean page.
                    //
                    std::lock_guard lock{g_eptMutex};
                    flipPage(ctx, PageT::clean);

                    if (::intel_x64::ept::pt::entry::execute_access::is_enabled(ctx->pte))
                        bferror_nhex(0, "write: exec not disabled", gpa4k);
                    if (::intel_x64::ept::pt::entry::read_access::is_disabled(ctx->pte))
                        bferror_nhex(0, "write: read not enabled", gpa4k);
                    if (::intel_x64::ept::pt::entry::write_access::is_disabled(ctx->pte))
                        bferror_nhex(0, "write: write not enabled", gpa4k);
                }
            }
            else
            {
                bfalert_nhex(ALERT_LVL, "write: different cr3", cr3);

                // Deactivate the split context, since it seems like the
                // application changed.
                //
                deactivateSplitImpl(ctx);
            }
        } else {
            bfalert_nhex(ALERT_LVL, "write: no split context found", gpa4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_mainMap.entry(gpa4k);

            // Check page granularity.
            //
            if (g_mainMap.is_4k(gpa4k)) {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!write_access::is_enabled(entry))
                {
                    bfalert_nhex(ALERT_LVL, "write: resetting access bits for 4k entry", gpa4k);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            } else {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!write_access::is_enabled(entry))
                {
                    const auto gpa2m = bfn::upper(gpa4k, ::intel_x64::ept::pt::from);
                    bfalert_nhex(ALERT_LVL, "write: resetting access bits for 2m entry", gpa2m);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            }
        }

        info.ignore_advance = true;
        return true;
    }

    bool ept_execute_violation_handler(gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info) {
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "exec: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa4k", gpa4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (auto* ctx = g_splits.getContext(gpa4k); ctx != nullptr)
        {
            // Check if the exec violation occurred in the same CR3 context.
            //
            if (ctx->cr3 == cr3) {
                // Check for thrashing.
                //
                if (vmcs->save_state()->rip == m_prevRip) { m_ripCounter++; }
                else { m_prevRip = vmcs->save_state()->rip; m_ripCounter = 1; }

                if (m_ripCounter > 4)
                {
                    bfalert_nhex(THRASHING_LVL, "exec: thrashing detected", m_prevRip);
                    m_ripCounter = 1;

                    // Enable monitor trap flag for single stepping.
                    //
                    m_trapCtx = ctx;
                    eapis()->set_eptp(g_trapMap);
                    // writePte(ctx->pte.get(), ctx->cleanPhys, AccessBitsT::all);
                    // eapis()->enable_monitor_trap_flag();
                }
                else
                {
                    // Flip to shadow page.
                    //
                    std::lock_guard lock{g_eptMutex};
                    flipPage(ctx, PageT::shadow);

                    if (::intel_x64::ept::pt::entry::execute_access::is_disabled(ctx->pte))
                        bferror_nhex(0, "exec: exec not enabled", gpa4k);
                    if (::intel_x64::ept::pt::entry::read_access::is_enabled(ctx->pte))
                        bferror_nhex(0, "exec: read not disabled", gpa4k);
                    if (::intel_x64::ept::pt::entry::write_access::is_enabled(ctx->pte))
                        bferror_nhex(0, "exec: write not disabled", gpa4k);
                }
            }
            else
            {
                bfalert_nhex(ALERT_LVL, "exec: different cr3", cr3);

                // Deactivate the split context, since it seems like the
                // application changed.
                //
                deactivateSplitImpl(ctx);
            }
        } else {
            bfalert_nhex(ALERT_LVL, "exec: no split context found", gpa4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_mainMap.entry(gpa4k);

            // Check page granularity.
            //
            if (g_mainMap.is_4k(gpa4k)) {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!execute_access::is_enabled(entry))
                {
                    bfalert_nhex(ALERT_LVL, "exec: resetting access bits for 4k entry", gpa4k);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            } else {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!execute_access::is_enabled(entry))
                {
                    const auto gpa2m = bfn::upper(gpa4k, ::intel_x64::ept::pt::from);
                    bfalert_nhex(ALERT_LVL, "exec: resetting access bits for 2m entry", gpa2m);
                    std::lock_guard lock{g_eptMutex};
                    writePte(entry.get(), phys_addr::get(entry), AccessBitsT::all);
                }
            }
        }

        info.ignore_advance = true;
        return true;
    }

    // -----------------------------------------------------------------------------
    // Monitor Trap Handler
    // -----------------------------------------------------------------------------

    bool monitor_trap_handler(
        gsl::not_null<vmcs_t *> vmcs, monitor_trap_handler::info_t &info)
    {
        bfignored(vmcs);
        bfignored(info);

        bfdebug_info(MONITOR_TRAP_LVL, "monitor trap");

        // Flip back to shadow page.
        //
        // flipPage(m_trapCtx, PageT::shadow);
        eapis()->set_eptp(g_mainMap);
        ::intel_x64::vmx::invept_global();

        return true;
    }

private:

    // This function returns 1 to the caller
    // to indicate that the HV is present.
    //
    uint64_t hvPresent() const noexcept { return 1ull; }

    // This function returns 1 to the caller
    // to indicate that the split got activated.
    //
    // Nothing is actually done here, since the splits will
    // be directly actived upon creation. It is here for
    // legacy purposes.
    //
    uint64_t activateSplit() const noexcept { return 1ull; }

    // This function will create a split context for the
    // provided GVA and CR3.
    //
    // If needed, the affected range will be remapped to 4k
    // granularity to minimize the EPT violation exits we will
    // get.
    //
    uint64_t createSplitContext(const uintptr_t gva, const uint64_t cr3) {
        // Check whether there already is a split for
        // the relevant 4k page.
        //
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);
        if (auto* ctx = g_splits.getContext(gpa4k); ctx != nullptr) {
            // Since we already have a split for the requested gva,
            // we will just increase the refCounter.
            //
            g_splits.incCounter(ctx);

            bfdebug_transaction(DEBUG_LVL, [&](std::string* msg) {
                bfdebug_nhex(0, "create: increased refCount", gpa4k, msg);
                bfdebug_subndec(0, "refCount", ctx->refCount, msg);
            });
        } else {
            // We don't seem to have a split for the requested gva,
            // check whether we have to remap the relevant 2m page range.
            //
            if (!g_mainMap.is_4k(gpa4k)) {
                const auto gpa2m = bfn::upper(gpa4k, ::intel_x64::ept::pd::from);
                bfdebug_nhex(DEBUG_LVL, "create: remapping 2m page range to 4k", gpa2m);
                // We need to remap the relevant 2m page range
                // to 4k granularity.
                //
                std::lock_guard lock{g_eptMutex};
                ept::identity_map_convert_2m_to_4k(g_mainMap, gpa2m);
                ::intel_x64::vmx::invept_global();  // necessary?
            } else {
                bfdebug_nhex(0, "create: already remapped", gpa4k);
            }

            // Ask for a free context which we can configure.
            //
            if (auto* ctx = g_splits.getFreeContext(); ctx != nullptr) {
                bfdebug_nhex(DEBUG_LVL, "create: creating split context", gpa4k);

                // Allocate memory for the shadow page.
                //
                ctx->shadowPage = std::make_unique<uint8_t[]>(::intel_x64::ept::pt::page_size);
                ctx->shadowVirt = reinterpret_cast<uintptr_t>(ctx->shadowPage.get());
                ctx->shadowPhys = g_mm->virtint_to_physint(ctx->shadowVirt);

                // Map clean page into VMM (Host) memory.
                //
                const auto vmmData =
                bfvmm::x64::make_unique_map<uint8_t>(
                    gva4k,
                    cr3,
                    ::intel_x64::ept::pt::page_size
                );

                // Copy contents of clean page (VMM copy) to shadow page.
                //
                std::memmove(
                    reinterpret_cast<void*>(ctx->shadowVirt),
                    reinterpret_cast<void*>(vmmData.get()),
                    ::intel_x64::ept::pt::page_size
                );

                // Insert other needed data.
                //
                ctx->cr3 = cr3;
                ctx->pte = g_mainMap.entry(gpa4k);
                ctx->cleanVirt = gva4k;
                ctx->cleanPhys = gpa4k;

                // Reset reference counter, flip to shadow page
                // and flush TLB.
                //
                g_splits.resetCounter(ctx);
                flipPage(ctx, PageT::shadow);
                ::intel_x64::vmx::invept_global();

                bfdebug_transaction(SPLIT_CONTEXT_LVL, [&](std::string* msg) {
                    bfdebug_info(0, "create: new split context", msg);
                    bfdebug_subnhex(0, "cleanPhys", gpa4k, msg);
                    bfdebug_subnhex(0, "cleanVirt", gva4k, msg);
                    bfdebug_subnhex(0, "shadowPhys", ctx->shadowPhys, msg);;
                    bfdebug_subnhex(0, "shadowVirt", ctx->shadowVirt, msg);
                    bfdebug_subndec(0, "refCount", ctx->refCount, msg)
                    bfdebug_subnhex(0, "cr3", cr3, msg);
                });
            } else {
                bferror_nhex(ERROR_LVL, "create: no free context available", gpa4k);
                return 0ull;
            }
        }

       return 1ull;
    }

    // This function will deactivate (and free) a split context
    // for the give guest physical address (4k aligned).
    //
    uint64_t deactivateSplitImpl(SplitPool::SplitContext* ctx) {
        // Check if the reference count is greater than 0.
        //
        if (g_splits.decCounter(ctx) > 0ull)
        { return 1ull; }

        bfdebug_transaction(SPLIT_CONTEXT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "deactivate: deactivating split context", msg);
            bfdebug_subnhex(0, "cleanPhys", ctx->cleanPhys, msg);
            bfdebug_subnhex(0, "cleanVirt", ctx->cleanVirt, msg);
            bfdebug_subnhex(0, "shadowPhys", ctx->shadowPhys, msg);;
            bfdebug_subnhex(0, "shadowVirt", ctx->shadowVirt, msg);
            bfdebug_subndec(0, "refCount", ctx->refCount, msg)
            bfdebug_subnhex(0, "cr3", ctx->cr3, msg);
        });

        bfdebug_nhex(DEBUG_LVL, "deactivate: deactivating split context", ctx->cleanPhys);
        // Since we are the last one to use the split context,
        // we should be able to safely disable this split context.
        // We will first flip to the clean page and then restore
        // pass-through access bits. Only then will we disable the
        // split context and deallocate the shadow page.
        //
        {
            std::lock_guard lock{g_eptMutex};
            writePte(ctx->pte.get(), ctx->cleanPhys, AccessBitsT::all);
            ::intel_x64::vmx::invept_global();
        }
        const auto gpa4k = ctx->cleanPhys;
        ctx->enabled = false;
        ctx->shadowPage = nullptr;

        // Check if there are any more splits for the relevant
        // 2m page range.
        //
        const auto gpa2mStart = bfn::upper(gpa4k, ::intel_x64::ept::pd::from);
        const auto gpa2mEnd = gpa2mStart + ::intel_x64::ept::pd::page_size;
        for (auto i = 0; i < MAX_SPLITS; ++i) {
            const auto* temp = g_splits.getContextByIndex(i);
            if (temp->enabled
                && temp->cleanPhys >= gpa2mStart
                && temp->cleanPhys < gpa2mEnd)
                return 1ull;
        }

        bfdebug_nhex(DEBUG_LVL, "deactivate: remapping 2m page range to 2m", gpa2mStart);
        // Since there are no more splits for the relevant 2m
        // page range, we will remap the range back from 4k to
        // 2m granularity.
        //
        {
            std::lock_guard lock{g_eptMutex};
            ept::identity_map_convert_4k_to_2m(g_mainMap, gpa2mStart);
            auto& entry = g_mainMap.entry(gpa2mStart);
            writePte(entry, ::intel_x64::ept::pd::entry::phys_addr::get(entry), AccessBitsT::all, pteMask2m);
            ::intel_x64::vmx::invept_global();
        }

        return 1ull;
    }

    // This is a proxy function, which calls the actual
    // implementation for deactivating a split, for the
    // VMCall interface.
    //
    uint64_t deactivateSplit(const uintptr_t gva, const uint64_t cr3) {
        // Check whether there is a split for the relevant 4k page.
        //
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);
        if (auto* ctx = g_splits.getContext(gpa4k); ctx != nullptr) {
            // Call implementation.
            //
            return deactivateSplitImpl(ctx);
        } else {
            bfalert_nhex(ALERT_LVL, "deactivate: no split context found", gpa4k);
            return 0ull;
        }
    }

    // This function will call the deactivation function for all
    // split contexts.
    //
    uint64_t deactivateAllSplits() {
        bfdebug_info(DEBUG_LVL, "deactivate: deactivating all split contexts");

        // Loop over all split contexts, until none
        // is enabled anymore.
        //
        SplitPool::SplitContext* temp = g_splits.getFirstEnabledContext();
        while (temp != nullptr) {
            deactivateSplitImpl(temp);
            temp = g_splits.getFirstEnabledContext();
        }

        return 1ull;
    }

    // This function returns 1, if there is a split for the passed
    // GVA, otherwise 0.
    //
    uint64_t isSplit(const uintptr_t gva, const uint64_t cr3) noexcept {
        const auto gva4k = bfn::upper(gva, ::intel_x64::ept::pt::from);
        const auto gpa4k = bfvmm::x64::virt_to_phys_with_cr3(gva4k, cr3);
        return g_splits.getContext(gpa4k) != nullptr;
    }

    // This function will copy the given memory range from the guest memory
    // to the shadow page (host memory).
    //
    uint64_t writeMemory(const uintptr_t srcGva, const uintptr_t dstGva, const size_t len, const uint64_t cr3) {
        const auto dstGva4k = bfn::upper(dstGva, ::intel_x64::ept::pt::from);
        const auto dstGpa4k = bfvmm::x64::virt_to_phys_with_cr3(dstGva4k, cr3);

        bfdebug_transaction(WRITE_LVL, [&](std::string* msg) {
            bferror_info(0, "writeMemory: arguments", msg);
            bferror_subnhex(0, "srcGva", srcGva, msg);
            bferror_subnhex(0, "dstGva", dstGva, msg);
            bferror_subndec(0, "len", len, msg);
        });

        // Check whether there is a split for the relevant 4k page.
        //
        if (auto* ctx = g_splits.getContext(dstGpa4k); ctx != nullptr) {
            // Check if we have to write to two consecutive pages.
            //
            const auto dstGvaEnd = dstGva + len - 1;
            const auto dstGvaEnd4k = bfn::upper(dstGvaEnd, ::intel_x64::ept::pt::from);
            if (dstGva4k == dstGvaEnd4k)
            {
                bfdebug_nhex(DEBUG_LVL, "writeMemory: writing to 1 page", dstGpa4k);

                // Calculate the offset from the start of the page.
                //
                const auto writeOffset = dstGva - dstGva4k;

                // Map the guest memory for the source GVA into host (VMM) memory.
                //
                const auto vmmData = bfvmm::x64::make_unique_map<uint8_t>(srcGva, cr3, len);

                // Copy contents from source (VMM copy) to shadow page.
                //
                std::memmove(
                    reinterpret_cast<void*>(ctx->shadowVirt + writeOffset),
                    reinterpret_cast<void*>(vmmData.get()),
                    len
                );

                #if DEBUG_LEVEL > WRITE_LVL
                const auto start = reinterpret_cast<uint8_t*>(ctx->shadowVirt + writeOffset);
                const std::vector<uint8_t> bytes{start, start + len};

                bfdebug_transaction(WRITE_LVL, [&](std::string* msg) {
                    bfdebug_info(0, "writeMemory: shadow page", msg);
                    for (auto i = 0; i < bytes.size(); ++i) {
                        bfdebug_subnhex(0, std::to_string(i).c_str(), bytes[i], msg);
                    }
                });
                #endif


            } else {
                bfdebug_nhex(DEBUG_LVL, "writeMemory: writing to 2 pages", dstGpa4k);

                // Check if we already have a split context for the second page.
                // If not, create one.
                //
                const auto dstGpaEnd4k = bfvmm::x64::virt_to_phys_with_cr3(dstGvaEnd4k, cr3);
                auto* ctx2 = g_splits.getContext(dstGpaEnd4k);
                if (ctx2 == nullptr) {
                    createSplitContext(dstGvaEnd4k, cr3);
                    if (ctx2 = g_splits.getContext(dstGpaEnd4k); ctx2 == nullptr) {
                        bferror_nhex(ERROR_LVL, "writeMemory: split for second page failed", dstGpaEnd4k);
                        return 0ull;
                    }
                }

                // Calculate the offset from the start of the page.
                //
                const auto writeOffset = dstGva - dstGva4k;

                // Calculate lengths for first and second page.
                //
                const auto firstLen = /*dstGvaEnd4k*/
                    dstGva4k + ::intel_x64::ept::pt::page_size - dstGva - 1;
                const auto secondLen = dstGvaEnd - dstGvaEnd4k;

                // Check if the sum adds up.
                //
                if (firstLen + secondLen != len) {
                    bfdebug_transaction(ERROR_LVL, [&](std::string* msg) {
                        bferror_info(0, "writeMemory: size does not add up", msg);
                        bferror_subndec(0, "firstLen", firstLen, msg);
                        bferror_subndec(0, "secondLen", secondLen, msg);
                        bferror_subndec(0, "len", len, msg);
                    });
                    return 0ull;
                }

                // Map the guest memory for the source GVA into host memory.
                //
                const auto vmmData = bfvmm::x64::make_unique_map<uint8_t>(srcGva, cr3, len);

                // Write to first page.
                //
                std::memmove(
                    reinterpret_cast<void*>(ctx->shadowVirt + writeOffset),
                    reinterpret_cast<void*>(vmmData.get()),
                    firstLen
                );

                // Write to second page.
                //
                std::memmove(
                    reinterpret_cast<void*>(ctx2->shadowVirt),
                    reinterpret_cast<void*>(vmmData.get() + firstLen + 1),
                    secondLen
                );
            }
        } else {
            bfalert_nhex(ALERT_LVL, "writeMemory: no split context found", dstGpa4k);
            return 0ull;
        }

        return 1ull;
    }

};

} // namespace ept_split

// -----------------------------------------------------------------------------
// vCPU Factory
// -----------------------------------------------------------------------------

namespace bfvmm
{

std::unique_ptr<vcpu>
vcpu_factory::make_vcpu(vcpuid::type vcpuid, bfobject *obj)
{
    bfignored(obj);
    return std::make_unique<ept_split::Vcpu>(vcpuid);
}

}
