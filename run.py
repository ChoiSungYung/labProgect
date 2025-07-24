import sys
import os
from streamlit.web import cli as stcli

def main():
    """
    Streamlit 애플리케이션을 실행하는 메인 함수.
    PyInstaller로 패키징된 환경과 일반 파이썬 환경 모두를 지원합니다.
    """
    # PyInstaller로 패키징되었을 때의 경로와 일반 실행 시의 경로를 모두 처리
    if getattr(sys, 'frozen', False):
        # 패키징된 경우, app.py는 임시 폴더(_MEIPASS)에 위치합니다.
        app_path = os.path.join(sys._MEIPASS, 'app.py')
    else:
        # 일반 실행의 경우, 현재 파일과 같은 디렉토리에 위치합니다.
        app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    # Streamlit을 프로그램적으로 실행합니다.
    # --server.runOnSave=false 옵션은 불필요한 재실행을 방지합니다.
    sys.argv = [
        "streamlit", "run", app_path,
        "--global.developmentMode", "false", # 개발자 모드 강제 비활성화
        "--server.port", "8501",
        "--server.runOnSave", "false"
    ]
    stcli.main()

if __name__ == "__main__":
    main() 