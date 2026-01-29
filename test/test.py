import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.services.file_service import extract_text_from_file


class MockFile:
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self.path = str(file_path)
        self._file_handle = None
        
        ext = os.path.splitext(self.name)[1].lower()
        if ext == '.pdf':
            self.type = "application/pdf"
        elif ext in ['.jpg', '.jpeg']:
            self.type = "image/jpeg"
        elif ext == '.png':
            self.type = "image/png"
        else:
            self.type = "unknown"
    
    def __str__(self):
        return self.path
    
    def __fspath__(self):
        return self.path
    
    def read(self, size=-1):
        if self._file_handle is None:
            self._file_handle = open(self.path, 'rb')
        return self._file_handle.read(size)
    
    def seek(self, pos):
        if self._file_handle is None:
            self._file_handle = open(self.path, 'rb')
        return self._file_handle.seek(pos)
    
    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


def test():
    pdf_file = Path(__file__).parent / "image to test" / "image1.png"
    
    if not pdf_file.exists():
        print(f"PDF file not found: {pdf_file}")
        print(f"Make sure the file exists at: {pdf_file.absolute()}")
        return
    
    print(f"Testing with file: {pdf_file}")
    
    try:
        mock_file = MockFile(str(pdf_file))
        
        print("Extracting text...")
        text = extract_text_from_file(mock_file)
        
        print(f"\nExtraction successful!")
        print(f"Extracted {len(text)} characters")
        print(f"\nFirst 500 characters:")
        print("-" * 50)
        print(text[:500])
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()