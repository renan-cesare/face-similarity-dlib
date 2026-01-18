import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np


DEFAULT_THRESHOLD = 0.6  # valor comum usado como referência para embeddings do dlib


def load_models(predictor_path: Path, encoder_path: Path):
    if not predictor_path.exists():
        raise FileNotFoundError(f"Arquivo do predictor não encontrado: {predictor_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Arquivo do encoder não encontrado: {encoder_path}")

    pose_predictor = dlib.shape_predictor(str(predictor_path))
    face_encoder = dlib.face_recognition_model_v1(str(encoder_path))
    detector = dlib.get_frontal_face_detector()
    return detector, pose_predictor, face_encoder


def encode_face(image_path: Path, detector, pose_predictor, face_encoder) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    faces = detector(image, 1)
    if len(faces) == 0:
        raise ValueError(f"Nenhum rosto encontrado: {image_path}")
    if len(faces) > 1:
        raise ValueError(f"Mais de um rosto encontrado (use imagem com 1 rosto): {image_path}")

    landmarks = pose_predictor(image, faces[0])
    face_chip = dlib.get_face_chip(image, landmarks)
    embedding = np.array(face_encoder.compute_face_descriptor(face_chip), dtype=np.float32)
    return embedding


def distance_between(image1: Path, image2: Path, detector, pose_predictor, face_encoder) -> float:
    e1 = encode_face(image1, detector, pose_predictor, face_encoder)
    e2 = encode_face(image2, detector, pose_predictor, face_encoder)
    return float(np.linalg.norm(e1 - e2))


def similarity_score(distance: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Converte uma distância em um "score" 0..100 baseado em um threshold.
    Importante: isso NÃO é uma probabilidade real, é só um indicador visual.
    """
    score = max(0.0, (1.0 - (distance / threshold))) * 100.0
    return float(score)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def compare_folders(folder_a: Path, folder_b: Path, detector, pose_predictor, face_encoder, threshold: float):
    images_a = list_images(folder_a)
    images_b = list_images(folder_b)

    if not images_a:
        raise ValueError(f"Nenhuma imagem encontrada em: {folder_a}")
    if not images_b:
        raise ValueError(f"Nenhuma imagem encontrada em: {folder_b}")

    distances = []
    scores = []

    for a in images_a:
        for b in images_b:
            d = distance_between(a, b, detector, pose_predictor, face_encoder)
            s = similarity_score(d, threshold=threshold)
            distances.append(d)
            scores.append(s)

    return float(np.mean(distances)), float(np.mean(scores)), len(images_a), len(images_b)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Projeto de estudo: comparar similaridade de rostos usando embeddings (dlib + OpenCV)."
    )

    parser.add_argument(
        "--predictor",
        type=Path,
        default=Path("models/shape_predictor_68_face_landmarks.dat"),
        help="Caminho para o shape_predictor_68_face_landmarks.dat",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=Path("models/dlib_face_recognition_resnet_model_v1.dat"),
        help="Caminho para o dlib_face_recognition_resnet_model_v1.dat",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold usado no score (valor comum: 0.6).",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_img = sub.add_parser("image", help="Compara 2 imagens (1 rosto por imagem).")
    p_img.add_argument("--img1", type=Path, required=True)
    p_img.add_argument("--img2", type=Path, required=True)

    p_fold = sub.add_parser("folders", help="Compara 2 pastas (todas as combinações).")
    p_fold.add_argument("--person_a", type=Path, required=True)
    p_fold.add_argument("--person_b", type=Path, required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    detector, pose_predictor, face_encoder = load_models(args.predictor, args.encoder)

    if args.mode == "image":
        d = distance_between(args.img1, args.img2, detector, pose_predictor, face_encoder)
        s = similarity_score(d, threshold=args.threshold)
        print(f"Distância (L2): {d:.4f}")
        print(f"Score de similaridade (0-100, heurístico): {s:.2f}")

    elif args.mode == "folders":
        mean_d, mean_s, na, nb = compare_folders(
            args.person_a, args.person_b, detector, pose_predictor, face_encoder, threshold=args.threshold
        )
        print(f"Imagens em A: {na} | Imagens em B: {nb}")
        print(f"Distância média (L2): {mean_d:.4f}")
        print(f"Score médio (0-100, heurístico): {mean_s:.2f}")


if __name__ == "__main__":
    main()
