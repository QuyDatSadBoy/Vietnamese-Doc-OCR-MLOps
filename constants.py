# constants.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 (fine-tuned on MC-OCR 2021)

# ---------------------------------------------------------------------------
# Vietnamese character dictionary (used for custom PaddleOCR dict file)
# ---------------------------------------------------------------------------

VIETNAMESE_CHARS = (
    "aáàảãạăắằẳẵặâấầẩẫậ"
    "bcdđeéèẻẽẹêếềểễệ"
    "ghiíìỉĩịklmn"
    "oóòỏõọôốồổỗộơớờởỡợ"
    "pqrstuúùủũụưứừửữự"
    "vxyýỳỷỹỵ"
    "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "BCDĐEÉÈẺẼẸÊẾỀỂỄỆ"
    "GHIÍÌỈĨỊ"
    "OÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "UÚÙỦŨỤƯỨỪỬỮỰ"
    "VXYÝỲỶỸỴ"
    "0123456789"
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

# ---------------------------------------------------------------------------
# PaddleOCR-compatible vocabulary
# ---------------------------------------------------------------------------
NUM_CLASSES: int = len(VIETNAMESE_CHARS) + 1  # +1 for CTC blank

CHAR_TO_IDX: dict = {ch: i + 1 for i, ch in enumerate(VIETNAMESE_CHARS)}
IDX_TO_CHAR: dict = {i + 1: ch for i, ch in enumerate(VIETNAMESE_CHARS)}

# ---------------------------------------------------------------------------
# PP-OCRv4 inference input dimensions
# Recognition model: fixed height=48, standard width=320
# ---------------------------------------------------------------------------
REC_IMAGE_HEIGHT: int = 48
REC_IMAGE_WIDTH: int = 320
REC_IMAGE_CHANNELS: int = 3   # RGB

# Detection model: multiples of 32
DET_SHORT_SIDE: int = 736

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "MC-OCR 2021"
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

