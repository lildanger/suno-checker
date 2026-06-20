# -*- mode: python ; coding: utf-8 -*-
import os
import onnxruntime

# 收集 onnxruntime/capi 下的所有原生二进制文件 (dll/pyd/so/dylib)
ort_capi_path = os.path.dirname(onnxruntime.capi.__file__)
ort_binaries = []
bin_exts = ('.dll', '.pyd', '.so', '.dylib')
if os.path.exists(ort_capi_path):
    for f in os.listdir(ort_capi_path):
        if f.endswith(bin_exts):
            ort_binaries.append((os.path.join(ort_capi_path, f), '.'))

a = Analysis(
    ['predict.py'],
    pathex=[],
    binaries=ort_binaries,
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
    name='suno-checker',
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
    name='suno-checker',
)
