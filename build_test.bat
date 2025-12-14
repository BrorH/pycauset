@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\ireal\Documents\pycauset\build"
cmake --build . --config Release --target test_symmetric_matrix
if %errorlevel% neq 0 exit /b %errorlevel%
ctest -C Release -R SymmetricMatrixTests --output-on-failure
