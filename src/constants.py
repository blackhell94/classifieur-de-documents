"""
Constantes partagées du pipeline - Mapping des 16 catégories de documents
"""

ID2LABEL = {
    0: "letter", 1: "form", 2: "email", 3: "handwritten",
    4: "advertisement", 5: "scientific report", 6: "scientific publication",
    7: "specification", 8: "file folder", 9: "news article",
    10: "budget", 11: "invoice", 12: "presentation",
    13: "questionnaire", 14: "resume", 15: "memo"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

NUM_CLASSES = 16

LABEL_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]
