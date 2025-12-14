@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
dumpbin /exports python\pycauset\pycauset_core.dll | findstr "set_scalar"
