import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        output_size = output_size or hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(gate * x + (1.0 - gate) * residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_variables, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.num_variables = num_variables
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_variables)
        ])
        self.selection_grn = GatedResidualNetwork(
            num_variables * input_size, hidden_size, num_variables, dropout
        )

    def forward(self, x):
        B, T, V, E = x.shape
        var_outputs = []
        for i in range(V):
            var_outputs.append(self.variable_grns[i](x.select(2, i)))
        var_outputs = torch.stack(var_outputs, dim=2)

        flattened = x.reshape(B, T, V * E)
        weights = torch.softmax(self.selection_grn(flattened), dim=-1).unsqueeze(-1)
        selected = torch.sum(weights * var_outputs, dim=2)
        return selected, weights.squeeze(-1)

class StaticVariableSelectionNetwork(nn.Module):
    def __init__(self, num_variables, embedding_size, hidden_size, dropout=0.1):
        super().__init__()
        self.num_variables = num_variables
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(embedding_size, hidden_size, hidden_size, dropout)
            for _ in range(num_variables)
        ])
        self.selection_grn = GatedResidualNetwork(
            num_variables * embedding_size, hidden_size, num_variables, dropout
        )

    def forward(self, x):
        B, V, E = x.shape
        var_outputs = []
        for i in range(V):
            var_outputs.append(self.variable_grns[i](x.select(1, i)))
        var_outputs = torch.stack(var_outputs, dim=1)

        flattened = x.reshape(B, V * E)
        weights = torch.softmax(self.selection_grn(flattened), dim=-1).unsqueeze(-1)
        selected = torch.sum(weights * var_outputs, dim=1)
        return selected, weights.squeeze(-1)

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        static_cardinalities,
        past_features=18,
        future_features=14,
        hidden_size=64,
        embedding_size=16,
        num_heads=4,
        dropout=0.1,
        forecast_horizon=7,
        num_quantiles=3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        self.num_quantiles = num_quantiles

        # Static Categorical Embeddings
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_size)
            for cardinality in static_cardinalities
        ])
        self.static_vsn = StaticVariableSelectionNetwork(
            len(static_cardinalities), embedding_size, hidden_size, dropout
        )

        # Temporal Linear Encoders
        self.past_variable_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(past_features)
        ])
        self.future_variable_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(future_features)
        ])

        # Temporal Variable Selection
        self.past_vsn = VariableSelectionNetwork(past_features, hidden_size, hidden_size, dropout)
        self.future_vsn = VariableSelectionNetwork(future_features, hidden_size, hidden_size, dropout)

        # Recurrent Sequence Processing (LSTM Encoder & Decoder)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Static Enrichment
        self.static_enrichment = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout)

        # Temporal Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Final Post-Attention GRN
        self.final_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Multi-Horizon Forecast Projection (Outputs quantiles e.g. P10, P50, P90 across all 7 days)
        self.forecast_head = nn.Linear(hidden_size, num_quantiles)

    def forward(self, past_x, future_x, static_x):
        # 1. Static processing
        static_emb = []
        for i, emb in enumerate(self.static_embeddings):
            static_emb.append(emb(static_x[:, i]))
        static_emb = torch.stack(static_emb, dim=1)
        static_context, _ = self.static_vsn(static_emb)

        # 2. Past temporal embedding & selection
        past_emb = []
        for i, emb in enumerate(self.past_variable_embeddings):
            past_emb.append(emb(past_x[:, :, i:i+1]))
        past_emb = torch.stack(past_emb, dim=2)
        past_selected, _ = self.past_vsn(past_emb)

        # 3. Future temporal embedding & selection
        future_emb = []
        for i, emb in enumerate(self.future_variable_embeddings):
            future_emb.append(emb(future_x[:, :, i:i+1]))
        future_emb = torch.stack(future_emb, dim=2)
        future_selected, _ = self.future_vsn(future_emb)

        # 4. LSTM sequence encoding/decoding
        _, (h, c) = self.encoder_lstm(past_selected)
        decoder_output, _ = self.decoder_lstm(future_selected, (h, c))

        # 5. Static enrichment
        static_expanded = static_context.unsqueeze(1).expand(-1, decoder_output.size(1), -1)
        enriched = self.static_enrichment(torch.cat([decoder_output, static_expanded], dim=-1))

        # 6. Multi-head Temporal Attention
        attn_out, _ = self.attention(enriched, enriched, enriched)
        attn_res = self.attention_norm(attn_out + enriched)

        # 7. Final processing & Multi-horizon Projection
        final_out = self.final_grn(attn_res)
        predictions = self.forecast_head(final_out)
        if self.num_quantiles == 1:
            predictions = predictions.squeeze(-1)

        return predictions
