"""
MurshidAI - Simple launcher script
This script helps you quickly start the backend and frontend servers.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("🎓 MurshidAI - مرشد")
    print("Bilingual AI Assistant for Saudi Scholarship Students")
    print("=" * 60)
    print()

    # Check if .env exists
    if not Path(".env").exists():
        print("⚠️  Warning: .env file not found!")
        print("Please copy .env.example to .env and configure it.")
        print()
        response = input("Create .env from .env.example now? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env created! Please edit it with your credentials.")
            print()
        return

    print("What would you like to do?")
    print()
    print("1. Run Backend (FastAPI)")
    print("2. Run Frontend (Streamlit)")
    print("3. Run Both (separate terminals required)")
    print("4. Ingest Telegram Data")
    print("5. Install Dependencies")
    print("6. Exit")
    print()

    choice = input("Enter your choice (1-6): ").strip()

    if choice == "1":
        print("\n🚀 Starting Backend...")
        os.chdir("backend")
        subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload"])

    elif choice == "2":
        print("\n🚀 Starting Frontend...")
        os.chdir("frontend")
        subprocess.run(["streamlit", "run", "streamlit_app.py"])

    elif choice == "3":
        print("\n⚠️  This requires two separate terminals!")
        print("\nTerminal 1 - Run this command:")
        print("cd backend && python -m uvicorn app.main:app --reload")
        print("\nTerminal 2 - Run this command:")
        print("cd frontend && streamlit run streamlit_app.py")

    elif choice == "4":
        print("\n📚 Starting data ingestion...")
        telegram_dir = input("Enter Telegram export directory path (default: ChatExport_2025-10-26): ").strip()
        if not telegram_dir:
            telegram_dir = "ChatExport_2025-10-26"

        os.chdir("backend")
        subprocess.run([sys.executable, "scripts/ingest_telegram.py", "--dir", f"../{telegram_dir}"])

    elif choice == "5":
        print("\n📦 Installing dependencies...")

        print("\n➡️  Installing backend dependencies...")
        os.chdir("backend")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        print("\n➡️  Installing frontend dependencies...")
        os.chdir("../frontend")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        print("\n✅ All dependencies installed!")

    elif choice == "6":
        print("\n👋 Goodbye!")
        return

    else:
        print("\n❌ Invalid choice!")

if __name__ == "__main__":
    main()