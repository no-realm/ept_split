#include <iostream>

#include <bfaffinity.h>
#include <intrinsics.h>

/*
 * VMCall Registers
 *
 * Defines a structure that stores each register. The register names are
 * generic so that they can be used by different CPU architectures.
 *
 * Intel: (unused: rsp, rbp, rdi)
 * r0 = rax
 * r1 = rdx
 * r2 = rcx
 * r3 = rbx
 * r4 = rsi
 * r5 = r8
 * r6 = r9
 * r7 = r10
 * r8 = r11
 * r9 = r12
 * r10 = r13
 * r11 = r14
 * r12 = r15
 * r13 = undefined
 * ...
 * r31 = undefined
 *
 * ARM:
 * <TBD>
 */
struct vmcall_registers_t
{
    uintptr_t r00;
    uintptr_t r01;
    uintptr_t r02;
    uintptr_t r03;
    uintptr_t r04;
    uintptr_t r05;
    uintptr_t r06;
    uintptr_t r07;
    uintptr_t r08;
    uintptr_t r09;
    uintptr_t r10;
    uintptr_t r11;
    uintptr_t r12;
    uintptr_t r13;
    uintptr_t r14;
    uintptr_t r15;
};

extern "C" void _platform_vmcall(struct vmcall_registers_t *regs) noexcept;

std::string testString = "hello world\n";

void
hello_world()
{ std::clog << testString; }

int main()
{
    // Set affinity.
    //
    // set_affinity(0);

    // Registers for VMCall
    //
    vmcall_registers_t regs;

    // Check if the HV is present.
    //
    regs.r02 = 0;
    _platform_vmcall(&regs);
    if (!regs.r02)
    {
        std::clog << "hv not present\n";
        return 1;
    }

    // Create split context.
    //
    regs.r02 = 1;
    regs.r03 = reinterpret_cast<uintptr_t>(hello_world);
    std::clog << "addr:" << std::hex << regs.r03 << '\n';
    _platform_vmcall(&regs);
    if (!regs.r02)
    {
        std::clog << "somthing went wrong\n";
        return 1;
    }

    regs.r02 = 2;
    regs.r03 = reinterpret_cast<uintptr_t>(hello_world);
    _platform_vmcall(&regs);
    if (!regs.r02)
    {
        std::clog << "somthing went wrong\n";
        return 1;
    }

    hello_world();
    auto* str = testString.data();
    str[0] = 'd';
    hello_world();

    regs.r02 = 4;
    _platform_vmcall(&regs);
    if (!regs.r02)
    {
        std::clog << "somthing went wrong\n";
        return 1;
    }

    hello_world();

    std::clog << '\n';
}
