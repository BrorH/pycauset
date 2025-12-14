@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
dumpbin /imports python\pycauset\_pycauset.pyd | findstr "pycauset_core.dll"
dumpbin /imports python\pycauset\_pycauset.pyd > imports.txt
