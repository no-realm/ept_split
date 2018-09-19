# EPT Splitting Extension

## Description

*TBD*
Using EPT to hook a function call is a common use case for reverse engineering,
introspection, and offensive / defensive research. This repository provides a
simple example of how this can be done using the [Bareflank Hypervisor](https://github.com/Bareflank/hypervisor),
and the [Extended APIs](https://github.com/Bareflank/extended_apis) repo using EPT.
For further information about the Bareflank Hypervisor and how to create extensions, please see the following
documentation.

## Compilation / Usage

To setup our extension, run the following (tested with Cygwin):

```
git clone https://github.com/Bareflank/hypervisor
git clone https://github.com/Bareflank/extended_apis
git clone https://github.com/Randshot/ept_split.git
mkdir build; cd build
cmake ../hypervisor -DDEFAULT_VMM=ept_split_vmm -DEXTENSION="../extended_apis;../ept_split"
make -j<# cores + 1>
```

To start the hypervisor, run the following commands:

```
make driver_quick
make quick
```

To run the monitor application, run the following commands:

```
make monitor
```

To stop and unload the hypervisor, run the following commands:

```
make unload
make driver_unload
```
