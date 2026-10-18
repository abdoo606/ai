@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title AI Browser Agent - التثبيت بنقرة واحدة
color 0A

echo ==============================================
echo    🤖 مساعد التصفح الذكي - سهل وسريع
echo ==============================================
echo.
echo ⏳ انتظر، كل شي بيصير أوتوماتيك.
echo    لا تغلق هذه النافذة!
echo.

cd /d "%USERPROFILE%"
set "WORKDIR=%USERPROFILE%\ai-browser-agent"
if not exist "%WORKDIR%" mkdir "%WORKDIR%"
cd /d "%WORKDIR%"

REM ========== 1) تحميل بايثون المحمول ==========
if exist "python\python.exe" goto PYTHON_OK
echo 📥 [1/5] تحميل بايثون...
curl -L -o python.zip https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
if not exist "python" mkdir python
tar -xf python.zip -C python
del python.zip
echo ⚙️  إعداد pip...
curl -L -o python\get-pip.py https://bootstrap.pypa.io/get-pip.py
python\python.exe python\get-pip.py --quiet
del python\get-pip.py
echo Lib\site-packages> python\python311._pth
echo .>> python\python311._pth
echo ..>> python\python311._pth
echo import site>> python\python311._pth
:PYTHON_OK
echo ✅ بايثون جاهز
echo.

REM ========== 2) تحميل cloudflared ==========
if exist "cloudflared.exe" goto CF_OK
echo 📥 [2/5] تحميل أداة النفق...
curl -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
:CF_OK
echo ✅ النفق جاهز
echo.

REM ========== 3) تثبيت المكتبات ==========
if exist "installed.flag" goto INSTALL_OK
echo 📦 [3/5] تثبيت المكتبات (3-5 دقائق، انتظر)...
python\python.exe -m pip install --upgrade pip --quiet
python\python.exe -m pip install gradio browser-use playwright python-dotenv langchain-google-genai langchain-openai --quiet
echo 🌐 تحميل كروم...
python\python.exe -m playwright install chromium
echo done> installed.flag
:INSTALL_OK
echo ✅ المكتبات جاهزة
echo.

REM ========== 4) كتابة تطبيق الوكيل ==========
echo 📝 [4/5] إعداد الواجهة...
(
echo import asyncio, os
echo import gradio as gr
echo from dotenv import load_dotenv
echo from browser_use import Agent
echo from langchain_google_genai import ChatGoogleGenerativeAI
echo from langchain_openai import ChatOpenAI
echo.
echo load_dotenv()
echo.
echo async def run_task(task, api_key, model_choice):
echo     if not task.strip():
echo         return "❌ اكتب المهمة أولًا."
echo     if not api_key.strip():
echo         return "❌ أدخل مفتاح API.\n\nاحصل عليه مجانًا من: https://aistudio.google.com/apikey"
echo     try:
echo         if "Gemini" in model_choice:
echo             llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
echo         else:
echo             llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
echo         agent = Agent(task=task, llm=llm)
echo         history = await agent.run()
echo         return history.final_result() or "✅ انتهت المهمة"
echo     except Exception as e:
echo         return f"❌ خطأ: {e}"
echo.
echo def run_sync(task, api_key, model_choice):
echo     return asyncio.run(run_task(task, api_key, model_choice))
echo.
echo with gr.Blocks(title="🤖 AI Browser Agent", theme=gr.themes.Soft(), css=".gr-button {font-size: 18px}") as demo:
echo     gr.Markdown("""# 🤖 مساعد التصفح الذكي
echo أكتب أي شي تبيه بالعربي والوكيل ينفذه في المتصفح.
echo
echo **أمثلة:**
echo - "اذهب إلى gmail.com وافتح إميلي"
echo - "سجل لي حساب في موقع تويتر"
echo - "ابحث في أمازون عن ساعة وأعطني السعر"
echo """)
echo     model_choice = gr.Dropdown(["Google Gemini (مجاني)", "OpenAI GPT-4o-mini"],
echo                                label="اختر المحرك", value="Google Gemini (مجاني)")
echo     api_key = gr.Textbox(label="🔑 مفتاح API (مجاني من Google)", type="password",
echo                         placeholder="AIzaSy...  (احصل عليه من aistudio.google.com/apikey)")
echo     task = gr.Textbox(label="📝 المهمة", lines=5,
echo                      placeholder="مثال: سجل لي حساب جديد في موقع ...")
echo     btn = gr.Button("🚀 ابدأ المهمة", variant="primary", size="lg")
echo     output = gr.Markdown(label="📊 النتيجة")
echo     btn.click(fn=run_sync, inputs=[task, api_key, model_choice], outputs=output)
echo.
echo if __name__ == "__main__":
echo     demo.queue().launch(server_name="0.0.0.0", server_port=7788)
) > agent_app.py

echo ✅ الواجهة جاهزة
echo.

REM ========== 5) التشغيل ==========
echo 🚀 [5/5] جاري التشغيل...
echo.

if not exist ".env" (
    echo GOOGLE_API_KEY=> .env
    echo ANONYMIZED_TELEMETRY=false>> .env
)

start "AI Agent" /B python\python.exe agent_app.py
timeout /t 8 /nobreak >nul

REM بدء النفق
set "CF_LOG=%TEMP%\cflog.txt"
del "%CF_LOG%" 2>nul
start "Cloudflare Tunnel" /B cloudflared.exe tunnel --url http://localhost:7788 --no-autoupdate > "%CF_LOG%" 2>&1

echo ⏳ انتظر 15 ثانية لإنشاء الرابط العام...
timeout /t 15 /nobreak >nul

cls
echo.
echo   ==============================================
echo      ✅ جاهز! افتح هذا الرابط في المتصفح:
echo   ==============================================
echo.
echo        http://localhost:7788
echo.
echo.

REM استخراج رابط النفق
powershell -NoProfile -Command "$c = Get-Content '%CF_LOG%' -Raw; $m = [regex]::Match($c, 'https://[a-z0-9-]+\.trycloudflare\.com'); if ($m.Success) { Write-Host ('   🌐 الرابط العام (أرسله لي لكي أتحكم به):'); Write-Host ('      ' + $m.Value) } else { Write-Host '   ⚠️  لم يتم إنشاء النفق بعد، أعد تشغيل الملف بعد قليل.' }"

echo.
echo   ==============================================
echo.
echo   📝 الخطوات المطلوبة منك:
echo   1. اذهب إلى  https://aistudio.google.com/apikey
echo   2. اضغط "Create API Key" وانسخ المفتاح
echo   3. الصق المفتاح في الواجهة
echo   4. اكتب المهمة بالعربي واضغط "ابدأ"
echo.
echo   لا تغلق هذه النافذة!
echo.
start http://localhost:7788
pause
