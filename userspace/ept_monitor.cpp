#include <iostream>

#include <bfaffinity.h>
#include <intrinsics.h>
#include <chrono>

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
    set_affinity(0);

    static const BYTE CODE[] = {
	  0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,    // mov    rax, 0x1
	  0xc3                                        // ret
	};

    // Registers for VMCall
    //
    vmcall_registers_t regs;

    regs.r02 = 1;
	regs.r03 = (uintptr_t)&CODE;
	_platform_vmcall(&regs);
	if (!regs.r02)
	{
        std::clog << "Something went wrong creating the shadow page";
		return 1;
	}

    regs.r02 = 6;
	regs.r03 = (uintptr_t)&CODE;
	regs.r04 = (uintptr_t)&CODE;
	regs.r05 = 0;
	_platform_vmcall(&regs);
	if (!regs.r02)
	{
        std::clog << "somthing went wrong writing the shadow page";
		return 1;
	}

    regs.r02 = 0;
	_platform_vmcall(&regs);
	if (!regs.r02)
	{
        std::clog << "hv not present";
		return 1;
	}

    // std::clog << "running map flipping performance tests\n";
    // auto times = 1000000;
    // auto begin = std::chrono::high_resolution_clock::now();

    // for (int a = 0; a < times; a++)
    // {
    //     regs.r02 = 12;
    //     _platform_vmcall(&regs);
    //     if (!regs.r02)
    //     {
    //         std::clog << "somthing went wrong\n";
    //         return 1;
    //     }
    // }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto dur = end - begin;
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    // auto ns = ms * 1000000;
    // auto nsPerIteration = ns / times;

    // std::clog << "ns per iteration: " << nsPerIteration << "\n";

    std::clog << '\n';
}
