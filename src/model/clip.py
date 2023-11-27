import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ImageEncoder, TextEncoder, ProjectionHead


class CLIP(nn.Module):
    def __init__(
        self,
        img_model_name: str,
        text_model_name: str,
        temperature: float,
        img_embedding: int,
        text_embedding: int,
        projection_dim: int,
        dropout: float,
        is_trainable: bool = True,
        use_pretrained: bool = True,
    ):
        super().__init__()

        self.img_model_name = img_model_name
        self.text_model_name = text_model_name
        self.temperature = temperature

        self.img_encoder = ImageEncoder(img_model_name, use_pretrained, is_trainable)
        self.text_encoder = TextEncoder(text_model_name, use_pretrained, is_trainable)

        self.img_projection = ProjectionHead(
            embedding_dim=img_embedding, projection_dim=projection_dim, dropout=dropout
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding, projection_dim=projection_dim, dropout=dropout
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        # img & text features from encoder
        img_features = self.img_encoder(batch["image"])
        text_features = self.text_encoder(batch["input_ids"], batch["attention_mask"])

        # img & text embedding from projection head
        img_embeddings = self.img_projection(img_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ img_embeddings.T) / self.temperature
        imgs_similarity = img_embeddings @ img_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax(
            (imgs_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction="none")
        imgs_loss = F.cross_entropy(logits.T, targets.T, reduction="none")
        loss = (imgs_loss + texts_loss) / 2.0

        return {
            "loss": loss.mean(),
            "img_embeddings": img_embeddings,
            "text_embeddings": text_embeddings,
        }
