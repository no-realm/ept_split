#include <bfcallonce.h>
#include <bfbitmanip.h>

#include <bfvmm/vcpu/vcpu_factory.h>
#include <bfvmm/memory_manager/arch/x64/unique_map.h>

#include <eapis/hve/arch/intel_x64/vcpu.h>
using namespace eapis::intel_x64;

#include "fplus/fplus.hpp"

#include <map>
#include <mutex>
#include <shared_mutex>

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
ept::mmap g_guest_map{};
ept::mmap g_copy_map{};
ept::mmap::entry_type g_dummy_pte{};

// Mutex objects for synchronizing access to the EPT
// and split context map.
//
std::mutex ept_mutex{};
std::shared_mutex split_mutex{};

// Helper macros for mutex management.
//
#define lock_guard(_mutex) std::lock_guard<std::mutex> lguard{_mutex}
#define shared_lock(_mutex) std::shared_lock<std::shared_mutex> lock{_mutex}
#define unique_lock(_mutex) std::unique_lock<std::shared_mutex> lock{_mutex}

// Macros for defining print levels.
//
#define VIOLATION_EXIT_LVL  1
#define SPLIT_CONTEXT_LVL   1
#define DEBUG_LVL           1
#define ALERT_LVL           0
#define ERROR_LVL           0

struct split_context
{
    // The following stores:
    // - Guest virtual address of the first split request
    // - CR3 of the requestee process
    // - Number of current modifications for this split
    // (- Indicates whether this split context is active) *removed
    // - Guest virtual address of the clean page
    // - Guest physical address of the clean page
    // - Host virtual address of the shadow page
    // - Host physical address of the shadow page
    //
    uint64_t requestee_gva{};
    uint64_t requestee_cr3{};
    std::atomic<size_t> use_count{};
    // bool active{};
    uint64_t clean_gva{};
    uint64_t clean_gpa{};
    uint64_t shadow_hva{};
    uint64_t shadow_hpa{};

    // This will hold the page table entry (PTE) that represents
    // this split. We will flip the access bits and change the
    // physical address this entry is pointing to, depending on
    // the EPT violation exit qualification.
    //
    std::reference_wrapper<ept::mmap::entry_type> pte{g_dummy_pte};

    // This is the owner (storage) of the shadow page.
    // We will allocate memory for it when a split is
    // requested.
    //
    std::unique_ptr<uint8_t[]> shadow_page{nullptr};
};

// Global map for storing all splits.
//
std::map<uint64_t, std::unique_ptr<split_context>> g_splits;

// Global map for keeping track of how many splits
// are currently active for a 2m entry.
// This will prevent us from accidentally remapping
// still active entries.
//
// std::unordered_map<uintptr_t, size_t> g_page_index;

// Mask for writing to PTEs (4k).
//
constexpr const auto pte_mask_4k = ::intel_x64::ept::pt::entry::phys_addr::mask
    | ::intel_x64::ept::pt::entry::read_access::mask
    | ::intel_x64::ept::pt::entry::write_access::mask
    | ::intel_x64::ept::pt::entry::execute_access::mask;

// Mask for writing to PTEs (4k).
//
constexpr const auto pte_mask_2m = ::intel_x64::ept::pd::entry::phys_addr::mask
    | ::intel_x64::ept::pd::entry::read_access::mask
    | ::intel_x64::ept::pd::entry::write_access::mask
    | ::intel_x64::ept::pd::entry::execute_access::mask;

class vcpu : public eapis::intel_x64::vcpu
{
    // Helper enum for page selection.
    //
    enum page
    {
        clean,
        shadow
    };

    // Helper enum for access bit selection.
    //
    enum access_bits
    {
        read        = 0b001,
        write       = 0b010,
        read_write  = 0b011,
        execute     = 0b100,
        all         = 0b111
    };

    // Helper function for writing to page table entries (PTE).
    //
    void write_pte(
        uintptr_t& pte, const uint64_t phys_addr, const access_bits bits, const uint64_t mask = pte_mask_4k) noexcept
    { pte = set_bits(pte, mask, phys_addr | bits); }

    // Helper function for flipping between pages
    //
    // Sets the physical address and access bits for
    // the provided split context.
    //
    void flip_page(
        const uint64_t gpa, const page flip_to) noexcept
    {
        shared_lock(split_mutex);
        if (flip_to == page::clean)
        { write_pte(g_splits[gpa]->pte.get(), g_splits[gpa]->clean_gpa, access_bits::read_write); }
        else
        { write_pte(g_splits[gpa]->pte.get(), g_splits[gpa]->shadow_hpa, access_bits::execute); }
    }

    // This stores the previous violation RIP value and
    // a counter for how many times it stayed the same.
    //
    uint64_t m_previous_rip{};
    unsigned m_rip_counter{};

public:

    // Constructor
    //
    // This is the only constructor the vCPU supports, so it must be
    // overloaded.
    //
    vcpu(vcpuid::type id) :
        eapis::intel_x64::vcpu{id}
    {
        using namespace vmcs_n;
        using mt_delegate_t = monitor_trap_handler::handler_delegate_t;
        using eptv_delegate_t = ept_violation_handler::handler_delegate_t;

        // Add VMCall handler.
        exit_handler()->add_handler(
            exit_reason::basic_exit_reason::vmcall,
            ::handler_delegate_t::create<vcpu, &vcpu::vmcall_handler>(this)
        );

        // Add Monitor Trap handler.
        //
        eapis()->add_monitor_trap_handler(
            mt_delegate_t::create<vcpu, &vcpu::monitor_trap_handler>(this)
        );

        // Add EPT violation handlers.
        //
        eapis()->add_ept_execute_violation_handler(
            eptv_delegate_t::create<vcpu, &vcpu::ept_execute_violation_handler>(this)
        );

        eapis()->add_ept_read_violation_handler(
            eptv_delegate_t::create<vcpu, &vcpu::ept_read_violation_handler>(this)
        );

        eapis()->add_ept_write_violation_handler(
            eptv_delegate_t::create<vcpu, &vcpu::ept_write_violation_handler>(this)
        );

        // Setup the EPT memory map once.
        //
        bfn::call_once(flag, [&] {
            lock_guard(ept_mutex);
            ept::identity_map(g_copy_map, MAX_PHYS_ADDR);
            ept::identity_map(g_guest_map, MAX_PHYS_ADDR);
        });

        eapis()->enable_vpid();
        eapis()->set_eptp(g_guest_map);
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
                    state->rcx = hv_present();
                    break;
                }

                // create_split_context(int_t gva)
                case 1ull:
                {
                    state->rcx = create_split_context(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // activate_split(int_t gva)
                case 2ull:
                {
                    state->rcx = activate_split();
                    break;
                }

                // deactivate_split(int_t gva)
                case 3ull:
                {
                    state->rcx = deactivate_split(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // deactivate_all_splits()
                case 4ull:
                {
                    state->rcx = deactivate_all_splits();
                    break;
                }

                // is_split(int_t gva)
                case 5ull:
                {
                    state->rcx = is_split(state->rbx, vmcs_n::guest_cr3::get());
                    break;
                }

                // write_to_shadow_page(int_t from_va, int_t to_va, size_t size)
                case 6ull:
                {
                    state->rcx = write_to_shadow_page(state->rbx, state->rbx, state->rsi, vmcs_n::guest_cr3::get());
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

    bool ept_read_violation_handler(
        gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info)
    {
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "read: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa", gpa, msg);
            bfdebug_subnhex(0, "gpa_4k", gpa_4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (shared_lock(split_mutex); !fplus::map_contains(g_splits, gpa_4k))
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "read: no split context found", gpa_4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_guest_map.entry(gpa);

            // Check page granularity.
            //
            if (lock_guard(ept_mutex); g_guest_map.is_4k(entry))
            {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!read_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "read: resetting access bits for 4k entry");
                    write_pte(entry.get(), phys_addr::get(entry), access_bits::all);
                }
            }
            else
            {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!read_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "read: resetting access bits for 2m entry");
                    write_pte(entry.get(), phys_addr::get(entry), access_bits::all, pte_mask_2m);
                }
            }

            info.ignore_advance = true;
            return true;
        }

        // Check if the read violation occurred in the same CR3 context.
        //
        if (shared_lock(split_mutex); g_splits[gpa_4k]->requestee_cr3 == cr3)
        {
            lock.unlock();

            // Check for thrashing.
            //
            if (vmcs->save_state()->rip == m_previous_rip) { m_rip_counter++; }
            else { m_previous_rip = vmcs->save_state()->rip; m_rip_counter = 1; }

            if (m_rip_counter > 3)
            {
                bfalert_nhex(ALERT_LVL, "read: thrashing detected", m_previous_rip);
                m_rip_counter = 1;

                // Enable monitor trap flag for single stepping.
                //
                eapis()->set_eptp(g_copy_map);
                eapis()->enable_monitor_trap_flag();
            }
            else
            {
                // Flip to clean page.
                //
                lock_guard(ept_mutex);
                flip_page(gpa_4k, page::clean);
            }
        }
        else
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "read: different CR3 detected", cr3);

            // Deactivate the split context, since it seems like the
            // application changed.
            //
            deactivate_split_imlp(gpa_4k);

            // Temporary solution by just resetting the access bits
            // and setting the physical address to the clean page.
            //
            // write_pte(g_splits[gpa_4k]->pte.get(), g_splits[gpa_4k]->clean_gpa, access_bits::all);
        }

        info.ignore_advance = true;
        return true;
    }

    bool ept_write_violation_handler(
        gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info)
    {
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "write: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa", gpa, msg);
            bfdebug_subnhex(0, "gpa_4k", gpa_4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (shared_lock(split_mutex); !fplus::map_contains(g_splits, gpa_4k))
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "write: no split context found", gpa_4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_guest_map.entry(gpa);

            // Check page granularity.
            //
            if (lock_guard(ept_mutex); g_guest_map.is_4k(entry))
            {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!write_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "write: resetting access bits for 4k entry");
                    write_pte(entry.get(), phys_addr::get(entry), access_bits::all);
                }
            }
            else
            {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!write_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "write: resetting access bits for 2m entry");
                    write_pte(entry.get(), phys_addr::get(entry), access_bits::all, pte_mask_2m);
                }

            }

            info.ignore_advance = true;
            return true;
        }

        // Check if the write violation occurred in the same CR3 context.
        //
        if (shared_lock(split_mutex); g_splits[gpa_4k]->requestee_cr3 == cr3)
        {
            lock.unlock();

            // Check for thrashing.
            //
            if (vmcs->save_state()->rip == m_previous_rip) { m_rip_counter++; }
            else { m_previous_rip = vmcs->save_state()->rip; m_rip_counter = 1; }

            if (m_rip_counter > 3)
            {
                bfalert_nhex(ALERT_LVL, "write: thrashing detected", m_previous_rip);
                m_rip_counter = 1;

                // Enable monitor trap flag for single stepping.
                //
                eapis()->set_eptp(g_copy_map);
                eapis()->enable_monitor_trap_flag();
            }
            else
            {
                // Flip to clean page.
                //
                lock_guard(ept_mutex);
                flip_page(gpa_4k, page::clean);
            }
        }
        else
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "write: different CR3 detected", cr3);

            // Deactivate the split context, since it seems like the
            // application changed.
            //
            deactivate_split_imlp(gpa_4k);

            // Temporary solution by just resetting the access bits
            // and setting the physical address to the clean page.
            //
            // write_pte(g_splits[gpa_4k]->pte.get(), g_splits[gpa_4k]->clean_gpa, access_bits::all);
        }

        info.ignore_advance = true;
        return true;
    }

    bool ept_execute_violation_handler(
        gsl::not_null<vmcs_t *> vmcs, ept_violation_handler::info_t &info)
    {
        const auto cr3 = vmcs_n::guest_cr3::get();
        const auto gva = vmcs_n::guest_linear_address::get();
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);

        bfdebug_transaction(VIOLATION_EXIT_LVL, [&](std::string* msg) {
            bfdebug_info(0, "exec: violation", msg);
            bfdebug_subnhex(0, "rip", vmcs->save_state()->rip, msg);
            bfdebug_subnhex(0, "gva", gva, msg);
            bfdebug_subnhex(0, "gpa", gpa, msg);
            bfdebug_subnhex(0, "gpa_4k", gpa_4k, msg);
            bfdebug_subnhex(0, "cr3", cr3, msg);
        });

        // Check if there is a split context for the requested page.
        //
        if (shared_lock(split_mutex); !fplus::map_contains(g_splits, gpa_4k))
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "exec: no split context found", gpa_4k);

            // Get page table entry.
            //
            std::reference_wrapper<ept::mmap::entry_type> entry = g_guest_map.entry(gpa);

            // Check page granularity.
            //
            if (lock_guard(ept_mutex); g_guest_map.is_4k(entry))
            {
                using namespace ::intel_x64::ept::pt::entry;

                // Check for outdated access bits.
                //
                if (!execute_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "exec: resetting access bits for 4k entry");
                    write_pte(entry.get(), phys_addr::get(entry), access_bits::all);
                }
            }
            else
            {
                using namespace ::intel_x64::ept::pd::entry;

                // Check for outdated access bits.
                //
                if (!execute_access::is_enabled(entry))
                {
                    bferror_info(ERROR_LVL, "exec: resetting access bits for 2m entry");
                write_pte(entry.get(), phys_addr::get(entry), access_bits::all, pte_mask_2m);
                }
            }

            info.ignore_advance = true;
            return true;
        }

        // Check if the execute violation occurred in the same CR3 context.
        //
        if (shared_lock(split_mutex); g_splits[gpa_4k]->requestee_cr3 == cr3)
        {
            lock.unlock();

            // Check for thrashing.
            //
            if (vmcs->save_state()->rip == m_previous_rip) { m_rip_counter++; }
            else { m_previous_rip = vmcs->save_state()->rip; m_rip_counter = 1; }

            if (m_rip_counter > 3)
            {
                bfalert_nhex(ALERT_LVL, "exec: thrashing detected", m_previous_rip);
                m_rip_counter = 1;

                // Enable monitor trap flag for single stepping.
                //
                eapis()->set_eptp(g_copy_map);
                eapis()->enable_monitor_trap_flag();
            }
            else
            {
                // Flip to clean page.
                //
                lock_guard(ept_mutex);
                flip_page(gpa_4k, page::shadow);
            }
        }
        else
        {
            lock.unlock();

            bfalert_nhex(ALERT_LVL, "exec: different CR3 detected", cr3);

            // Deactivate the split context, since it seems like the
            // application changed.
            //
            deactivate_split_imlp(gpa_4k);

            // Temporary solution by just resetting the access bits
            // and setting the physical address to the clean page.
            //
            // write_pte(g_splits[gpa_4k]->pte.get(), g_splits[gpa_4k]->clean_gpa, access_bits::all);
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

        bfdebug_info(DEBUG_LVL, "monitor trap");
        eapis()->set_eptp(g_guest_map);
        ::intel_x64::vmx::invept_global();

        return true;
    }

private:

    // This function returns 1 to the caller
    // to indicate that the HV is present.
    //
    uint64_t hv_present() const noexcept
    { return 1ull; }

    // This function returns 1 to the caller
    // to indicate that the split got activated.
    //
    // Nothing is actually done here, since the splits will
    // be directly actived upon creation. It is here for
    // legacy purposes.
    //
    uint64_t activate_split() const noexcept
    { return 1ull; }

    // This function will create a split context for the
    // provided GVA and CR3.
    //
    // If needed, the affected range will be remapped to 4k
    // granularity to minimize the EPT violation exits we will
    // get.
    //
    uint64_t create_split_context(
        const uint64_t gva, const uint64_t cr3)
    {
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);

        // Check whether we need to remap the relevant page range to 4k granularity.
        //
        const auto gpa_2m = bfn::upper(gpa, ::intel_x64::ept::pd::from);
        bool remapped = false;
        if (lock_guard(ept_mutex); g_guest_map.is_2m(gpa_2m))
        {
            bfdebug_nhex(DEBUG_LVL, "create: remapping 2m page to 4k", gpa_2m);

            // Remap identity map for the affected region from 2m
            // to 4k granularity.
            //
            ept::identity_map_convert_2m_to_4k(
                g_guest_map,
                gpa_2m
            );

            // Indicate that we remapped the affected region,
            // which means that we definitly don't already
            // have a split context for the requested GVA.
            //
            remapped = true;
            ::intel_x64::vmx::invept_global();
        }

        // Check whether there already is a split for
        // the relevant 4k page.
        //
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);
        if (shared_lock(split_mutex); remapped || !fplus::map_contains(g_splits, gpa_4k))
        {
            lock.unlock();

            bfdebug_nhex(DEBUG_LVL, "create: creating split context", gpa_4k);

            // Create new split context
            //
            {
                unique_lock(split_mutex);
                g_splits[gpa_4k] = std::make_unique<split_context>();
                g_splits[gpa_4k]->requestee_gva = gva;
                g_splits[gpa_4k]->requestee_cr3 = cr3;
                g_splits[gpa_4k]->clean_gva = bfn::upper(gva, ::intel_x64::ept::pt::from);
                g_splits[gpa_4k]->clean_gpa = gpa_4k;
                {
                    lock_guard(ept_mutex);
                    g_splits[gpa_4k]->pte = g_guest_map.entry(gpa);
                }

                // Allocate memory for the shadow page.
                //
                g_splits[gpa_4k]->shadow_page = std::make_unique<uint8_t[]>(::intel_x64::ept::pt::page_size);
                g_splits[gpa_4k]->shadow_hva = reinterpret_cast<uintptr_t>(g_splits[gpa_4k]->shadow_page.get());
                g_splits[gpa_4k]->shadow_hpa = g_mm->virtint_to_physint(g_splits[gpa_4k]->shadow_hva);
            }

            // Map data page into VMM (Host) memory.
            //
            lock.lock();
            const auto vmm_data =
                bfvmm::x64::make_unique_map<uint8_t>(
                    g_splits[gpa_4k]->clean_gva,
                    cr3,
                    ::intel_x64::ept::pt::page_size
                );

            // Copy contents of data page (VMM copy) to shadow page.
            //
            std::memmove(
                reinterpret_cast<void*>(g_splits[gpa_4k]->shadow_hva),
                reinterpret_cast<void*>(vmm_data.get()),
                ::intel_x64::ept::pt::page_size
            );
            lock.unlock();

            // Flip to shadow page and flush TLB.
            //
            bfdebug_nhex(DEBUG_LVL, "create: flipping to shadow page", gpa_4k);
            lock_guard(ept_mutex);
            flip_page(gpa_4k, page::shadow);
            ::intel_x64::vmx::invept_global();
        }

        // Increase use counter.
        //
        {
            unique_lock(split_mutex);
            g_splits[gpa_4k]->use_count++;
        }

        bfdebug_transaction(SPLIT_CONTEXT_LVL, [&](std::string* msg) {
            shared_lock(split_mutex);
            bfdebug_info(0, "create: split context", msg);
            bfdebug_subnhex(0, "gpa_4k", gpa_4k, msg);
            bfdebug_subndec(0, "use_count", g_splits[gpa_4k]->use_count, msg);
            bfdebug_subnhex(0, "shadow_hva", g_splits[gpa_4k]->shadow_hva, msg);
            bfdebug_subnhex(0, "shadow_hpa", g_splits[gpa_4k]->shadow_hpa, msg);
            bfdebug_subnhex(0, "requestee_gva", gva, msg);
            bfdebug_subnhex(0, "requestee_cr3", cr3, msg);
        });

       return 1ull;
    }

    // This is a proxy function, which calls the actual
    // implementation for deactivating a split, for the
    // VMCall interface.
    //
    uint64_t deactivate_split(
        const uint64_t gva, const uint64_t cr3)
    {
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);

        // Check whether there is a split for the relevant 4k page.
        //
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);
        if (shared_lock(split_mutex); !fplus::map_contains(g_splits, gpa_4k))
        {
            lock.unlock();
            bfalert_nhex(ALERT_LVL, "deactivate: no split context found", gpa_4k);
            return 0ull;
        }

        // Call implementation.
        //
        return deactivate_split_imlp(gpa_4k);
    }

    // This function will deactivate (and free) a split context
    // for the give guest physical address (4k aligned).
    //
    // This function will assume that a split context for the
    // given GPA is present!
    //
    uint64_t deactivate_split_imlp(
        const uint64_t gpa_4k)
    {
        bfdebug_transaction(SPLIT_CONTEXT_LVL, [&](std::string* msg) {
            shared_lock(split_mutex);
            bfdebug_info(0, "deactivate: split context", msg);
            bfdebug_subnhex(0, "gpa_4k", gpa_4k, msg);
            bfdebug_subndec(0, "use_count", g_splits[gpa_4k]->use_count, msg);
            bfdebug_subnhex(0, "shadow_hva", g_splits[gpa_4k]->shadow_hva, msg);
            bfdebug_subnhex(0, "shadow_hpa", g_splits[gpa_4k]->shadow_hpa, msg);
            bfdebug_subnhex(0, "requestee_gva", g_splits[gpa_4k]->requestee_gva, msg);
            bfdebug_subnhex(0, "requestee_cr3", g_splits[gpa_4k]->requestee_cr3, msg);
        });

        // Check the (new) use count of the split context.
        // If it is greater than 0, we will just return.
        //
        if (unique_lock(split_mutex); --g_splits[gpa_4k]->use_count > 0)
        { return 1ull; }

        // Since we are the last one to use the split context,
        // we should be able to safely free this split context.
        // We will first flip to the clean page and then restore
        // pass-through access bits. Only the will we erase the
        // split context object from the map.
        //
        {
            shared_lock(split_mutex);
            lock_guard(ept_mutex);
            write_pte(g_splits[gpa_4k]->pte.get(), g_splits[gpa_4k]->clean_gpa, access_bits::all);
            ::intel_x64::vmx::invept_global();
            g_splits.erase(gpa_4k);
        }

        // Check if there are any more splits for the relevant
        // 2m page range.
        //
        const auto gpa_2m_start = bfn::upper(gpa_4k, ::intel_x64::ept::pd::from);
        const auto gpa_2m_end = gpa_2m_start + ::intel_x64::ept::pd::page_size;
        {
            shared_lock(split_mutex);
            for (const auto& it : g_splits)
            {
                if (it.first >= gpa_2m_start && it.first < gpa_2m_end)
                { return 1ull; }
            }
        }

        // Since there are no more splits for the relevant 2m
        // page range, we will remap the range back from 4k to
        // 2m granularity.
        //
        bfdebug_nhex(DEBUG_LVL, "deactivate: remapping 4k page range to 2m", gpa_2m_start);
        lock_guard(ept_mutex);
        ept::identity_map_convert_4k_to_2m(g_guest_map, gpa_2m_start);
        auto& entry = g_guest_map.entry(gpa_2m_start);
        write_pte(entry, ::intel_x64::ept::pd::entry::phys_addr::get(entry), access_bits::all, pte_mask_2m);
        ::intel_x64::vmx::invept_global();

        return 1ull;
    }

    // This function will call the deactivation function for all
    // split contexts.
    //
    uint64_t deactivate_all_splits()
    {
        bfdebug_info(DEBUG_LVL, "deactivate: deactivating all split contexts");

        for (const auto& it : g_splits)
        { deactivate_split_imlp(it.first); }

        return 1ull;
    }

    // This function returns 1, if there is a split for the passed
    // GVA, otherwise 0.
    //
    uint64_t is_split(
        const uint64_t gva, const uint64_t cr3)
    {
        const auto gpa = bfvmm::x64::virt_to_phys_with_cr3(gva, cr3);
        const auto gpa_4k = bfn::upper(gpa, ::intel_x64::ept::pt::from);
        shared_lock(split_mutex);
        return fplus::map_contains(g_splits, gpa_4k) ? 1ull : 0ull;
    }

    // This function will copy the given memory range from the guest memory
    // to the shadow page (host memory).
    //
    uint64_t write_to_shadow_page(
        const uint64_t src_gva, const uint64_t dst_gva, const size_t len, const uint64_t cr3)
    {
        const auto dst_gpa = bfvmm::x64::virt_to_phys_with_cr3(dst_gva, cr3);

        // Check whether there is a split for the relevant 4k page.
        //
        const auto dst_gpa_4k = bfn::upper(dst_gpa, ::intel_x64::ept::pt::from);
        if (shared_lock(split_mutex); !fplus::map_contains(g_splits, dst_gpa_4k))
        {
            lock.unlock();
            bfalert_nhex(ALERT_LVL, "write_page: no split context found", dst_gpa_4k);
            return 0ull;
        }

        // Check if we have to write to two consecutive pages.
        //
        const auto dst_gva_4k = bfn::upper(dst_gva, ::intel_x64::ept::pt::from);
        const auto dst_end_gva = dst_gva + len - 1;
        const auto dst_end_gva_4k = bfn::upper(dst_end_gva, ::intel_x64::ept::pt::from);
        if (dst_gva_4k == dst_end_gva_4k)
        {
            bfdebug_nhex(DEBUG_LVL, "write_page: writing to 1 page", dst_gpa_4k);

            // Calculate the offset from the page start.
            //
            const auto write_offset = dst_gva - dst_gva_4k;

            // Map the guest memory for the source GVA into host memory.
            //
            const auto vmm_data = bfvmm::x64::make_unique_map<uint8_t>(src_gva, cr3, len);

            // Copy contents from source to shadow page.
            //
            shared_lock(split_mutex);
            std::memmove(
                reinterpret_cast<void*>(g_splits[dst_gpa_4k]->shadow_hva + write_offset),
                reinterpret_cast<void*>(vmm_data.get()),
                len
            );
        }
        else
        {
            bfdebug_nhex(DEBUG_LVL, "write_page: writing to 2 pages", dst_gpa_4k);

            // Check if we already have a split context for the second page.
            // If now, create one.
            //
            const auto dst_end_gpa_4k = dst_gpa_4k + ::intel_x64::ept::pt::page_size;
            if (shared_lock(split_mutex); !fplus::map_contains(g_splits, dst_end_gpa_4k))
            {
                lock.unlock();
                create_split_context(dst_gva + len, cr3);
            }

            // Calculate the offset from the page start.
            //
            const auto dst_gva_4k = bfn::upper(dst_gva, ::intel_x64::ept::pt::from);
            const auto write_offset = dst_gva - dst_gva_4k;

            // Calculate length for first and second page.
            //
            const auto first_len = dst_end_gva_4k - dst_gva - 1;
            const auto second_len = dst_end_gva - dst_end_gva_4k;

            // Check if the sum adds up.
            //
            if (first_len + second_len != len)
            {
                bfdebug_transaction(ERROR_LVL, [&](std::string* msg) {
                    bferror_info(0, "write_page: size does not add up", msg);
                    bferror_subndec(0, "first_len", first_len, msg);
                    bferror_subndec(0, "second_len", second_len, msg);
                    bferror_subndec(0, "len", len, msg);
                });
            }

            // Map the guest memory for the source GVA into host memory.
            //
            const auto vmm_data = bfvmm::x64::make_unique_map<uint8_t>(src_gva, cr3, len);

            shared_lock(split_mutex);

            // Write to first page.
            //
            std::memmove(
                reinterpret_cast<void*>(g_splits[dst_gpa_4k]->shadow_hva + write_offset),
                reinterpret_cast<void*>(vmm_data.get()),
                first_len
            );

            // Write to second page.
            //
            std::memmove(
                reinterpret_cast<void*>(g_splits[dst_end_gpa_4k]->shadow_hva),
                reinterpret_cast<void*>(vmm_data.get() + first_len + 1),
                second_len
            );
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
    return std::make_unique<ept_split::vcpu>(vcpuid);
}

}
