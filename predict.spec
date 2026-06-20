# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import onnxruntime

# 收集 onnxruntime/capi 下的所有原生二进制文件 (dll/pyd/so/dylib)
ort_capi_path = os.path.dirname(onnxruntime.capi.__file__)
ort_binaries = []
bin_exts = ('.dll', '.pyd', '.so', '.dylib')
if os.path.exists(ort_capi_path):
    for f in os.listdir(ort_capi_path):
        if f.endswith(bin_exts):
            # 1. 复制到 onnxruntime/capi 目录下，保持 Python 模块的包结构
            ort_binaries.append((os.path.join(ort_capi_path, f), 'onnxruntime/capi'))
            # 2. 如果是 DLL 依赖库，同时也复制到根目录下，方便系统加载器寻找
            if f.endswith(('.dll', '.so', '.dylib')):
                ort_binaries.append((os.path.join(ort_capi_path, f), '.'))

# 强制收集 Windows 下的 MSVC++ 核心运行时 DLLs，解决干净 Windows 环境下 DLL 初始化或找不到的崩溃问题
vc_binaries = []
if sys.platform == 'win32':
    sys32 = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32')
    vc_dlls = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']
    for dll in vc_dlls:
        dll_path = os.path.join(sys32, dll)
        if os.path.exists(dll_path):
            vc_binaries.append((dll_path, '.'))

is_mac = sys.platform == 'darwin'
app_name = 'Suno Checker' if is_mac else 'suno-checker'

a = Analysis(
    ['predict.py'],
    pathex=[],
    binaries=ort_binaries + vc_binaries,
    datas=[('ai_music_detector.onnx', '.')],
    hiddenimports=['fakeprint', 'scipy.ndimage._nd_image', 'scipy.ndimage._filters',
                   'scipy.special._ufuncs_cxx', 'soundfile', 'pyloudnorm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'onnxscript', 'sympy',
              'matplotlib', 'sklearn', 'pandas',
              'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineWidgets',
              'PyQt5.QtQml', 'PyQt5.QtSql', 'PyQt5.QtNetwork',
              'PyQt5.QtXml', 'PyQt5.QtSvg', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=app_name,
)

if is_mac:
    app = BUNDLE(
        coll,
        name='Suno Checker.app',
        icon=None,
        bundle_identifier='com.suno.checker',
    )

