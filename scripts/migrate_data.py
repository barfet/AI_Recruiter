import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Source directories
JOB_POSTINGS_DIR = PROJECT_ROOT / "data/job-postings"
RESUME_DIR = PROJECT_ROOT / "data/resume"

# Target directories
RAW_JOBS_DIR = PROJECT_ROOT / "data/raw/jobs"
RAW_RESUMES_DIR = PROJECT_ROOT / "data/raw/resumes"

def migrate_data():
    """Migrate data to new directory structure"""
    print("Starting data migration...")
    
    # Create target directories
    RAW_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Migrate job postings data
    if JOB_POSTINGS_DIR.exists():
        print("Migrating job postings data...")
        for item in JOB_POSTINGS_DIR.iterdir():
            target = RAW_JOBS_DIR / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
    
    # Migrate resume data
    if RESUME_DIR.exists():
        print("Migrating resume data...")
        for item in RESUME_DIR.iterdir():
            target = RAW_RESUMES_DIR / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
    
    print("Data migration completed!")

if __name__ == "__main__":
    migrate_data() 