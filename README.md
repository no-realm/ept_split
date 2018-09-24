# EPT Splitting Extension

## Description

*TBD*

## Compilation

To setup the extension, run the following (tested with Cygwin):

```
git clone -b ept_split https://github.com/Randshot/hypervisor
git clone -b ept_split https://github.com/Randshot/extended_apis
git clone https://github.com/Randshot/ept_split.git
mkdir build; cd build
cmake ../hypervisor -DDEFAULT_VMM=ept_split_vmm -DEXTENSION="../extended_apis;../ept_split"
make -j<# cores + 1>
```
*Note that the __ept_split__ branches point to a commit that is known to work with this extension.*

## Usage

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
