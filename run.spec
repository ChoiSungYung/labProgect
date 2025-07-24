# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

# 이전 datas 리스트는 주석 처리하거나 삭제합니다.
# datas = []
datas = copy_metadata('streamlit')
datas += [
    ('app.py', '.'),
    ('.venv/Lib/site-packages/streamlit/static', 'streamlit/static'),
    ('.venv/Lib/site-packages/plotly', 'plotly')
]

block_cipher = None


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=datas,  # 수정된 datas 리스트를 사용
    hiddenimports=['streamlit.runtime.scriptrunner.magic_funcs'], # 숨겨진 모듈 추가
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,         # 디버깅을 위해 True로 유지
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',      # 아이콘 경로 지정
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run',
)
