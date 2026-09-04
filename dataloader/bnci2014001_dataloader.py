"""BNCI2014001 MAT-file discovery and loading only."""
from pathlib import Path
from dataloader import load_mat_file

DATASET_NAME = "BNCI2014001"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / DATASET_NAME
SUBJECTS = tuple(range(1, 10))
SESSION_NAME_TO_ID = {"T": 1, "E": 2}

def session_id(session):
    if isinstance(session, str):
        session = session.upper()
        return int(session) if session.isdigit() else SESSION_NAME_TO_ID[session]
    return int(session)

def session_name(session):
    return {v: k for k, v in SESSION_NAME_TO_ID.items()}[session_id(session)]

def mat_path(data_path=DEFAULT_DATA_PATH, subject=1, session=1):
    return Path(data_path) / f"Data_S{int(subject):02d}_{session_name(session)}.mat"

def available_subjects(data_path=DEFAULT_DATA_PATH):
    return [s for s in SUBJECTS if any(mat_path(data_path, s, q).exists() for q in SESSION_NAME_TO_ID.values())]

def available_sessions(data_path=DEFAULT_DATA_PATH, subject=1):
    return [q for q in SESSION_NAME_TO_ID.values() if mat_path(data_path, subject, q).exists()]

def load_subject_session(data_path=DEFAULT_DATA_PATH, subject=1, session=1):
    session = session_id(session)
    return load_mat_file(mat_path(data_path, subject, session))
