# Model Card — Telecom Churn Prediction MLP

## Informações do Modelo

| Campo | Valor |
|---|---|
| Nome | Churn Prediction MLP |
| Versão | v1.0 |
| Tipo | Classificação Binária |
| Arquitetura | MLP — 30 → 128 → 64 → 1 (BatchNorm + ReLU + Dropout 0.3) |
| Framework | PyTorch 2.2 |
| Pipeline de dados | scikit-learn ColumnTransformer |
| API | FastAPI 0.111 |

---

## Uso Pretendido

**Contexto**: Empresa de telecomunicações deseja identificar clientes com alto risco de cancelamento para acionar ações proativas de retenção antes que o churn ocorra.

**Usuários pretendidos**: Times de retenção de clientes, CRM, sistemas automatizados de campanhas.

**Usos fora do escopo**:
- Não deve ser usado para discriminar clientes em condições de serviço ou precificação
- Não adequado para contextos geográficos muito diferentes do dataset de treino (origem norte-americana)

---

## Dados de Treinamento

| Campo | Detalhes |
|---|---|
| Dataset | Telco Customer Churn (IBM Sample Dataset) |
| Fonte | Kaggle — `blastchar/telco-customer-churn` |
| Registros | 7.043 |
| Features | 19 (após remoção do `customerID`) |
| Target | `Churn` (Yes=1 / No=0) |
| Desbalanceamento | ~26% positivos / ~74% negativos (razão 2.7:1) |
| Divisão | 68% treino / 17% validação / 15% teste holdout |
| Validação baselines | StratifiedKFold (k=5) |

---

## Performance

### Conjunto de Teste Holdout (threshold = 0.5)

| Modelo | AUC-ROC | PR-AUC | F1-Score | Accuracy |
|---|---|---|---|---|
| DummyClassifier | 0.500 | 0.265 | 0.000 | 0.735 |
| Logistic Regression | **0.850** | **0.637** | **0.624** | **0.740** |
| **MLP (PyTorch)** | 0.846 | 0.635 | 0.617 | 0.727 |

> **Nota**: MLP e Logistic Regression apresentam performance equivalente — resultado esperado para datasets tabulares pequenos (~7k registros). O diferencial do MLP é a flexibilidade de arquitetura para escalar com mais dados.

### MLP com Threshold Ótimo (0.34)

| Métrica | Valor |
|---|---|
| Recall churn | **95.4%** |
| Precisão churn | 42.8% |
| F1-Score | 0.591 |
| Redução de custo estimada | 11.5% vs. threshold 0.5 |

---

## Threshold e Decisão de Negócio

- **Threshold selecionado**: **0.34**
- **Critério**: minimização do custo combinado de falsos positivos e falsos negativos
- **Hipóteses de custo**:
  - Falso Negativo (churn não detectado): R$ 50 — perda estimada de receita mensal
  - Falso Positivo (intervenção desnecessária): R$ 10 — custo operacional da ação
- **Resultado**: threshold 0.34 reduz o custo total em ~11.5% em relação ao threshold padrão 0.5
- **Trade-off aceito**: menor precisão (42.8%) em favor de recall alto (95.4%) — mais adequado quando o custo de perder um cliente supera o custo de intervir desnecessariamente

---

## Limitações

1. **Data drift**: o modelo foi treinado em dados históricos. Performance pode degradar se o comportamento dos clientes mudar significativamente ao longo do tempo.
2. **Contexto geográfico**: o dataset é de origem norte-americana; padrões contratuais e de uso podem não se aplicar diretamente a outros mercados.
3. **Churn involuntário**: o modelo não distingue churn voluntário (decisão do cliente) de involuntário (inadimplência, suspensão por operadora).
4. **Features de comportamento ausentes**: dados de uso real (volume de chamadas, dados consumidos, tickets de suporte) não estão disponíveis no dataset e poderiam melhorar significativamente o modelo.
5. **Tamanho do dataset**: 7.043 registros é relativamente pequeno para redes neurais; um modelo linear (Logistic Regression) atinge performance equivalente.

---

## Vieses Conhecidos

1. **Clientes com baixo `tenure`**: o modelo tende a classificar clientes novos como alto risco, mesmo sem evidência clara de intenção de cancelamento.
2. **Correlação `TotalCharges` × `tenure`**: alta correlação entre as duas features pode criar redundância e amplificar vieses temporais.
3. **Desbalanceamento**: tratado via `pos_weight` dinâmico, mas grupos com características pouco representadas no treino podem ter performance inferior.

---

## Cenários de Falha

| Cenário | Impacto | Mitigação |
|---|---|---|
| Feature com valor inválido na API | HTTP 422 (Pydantic) | Validação automática no endpoint |
| Artefato de modelo ausente no startup | Servidor não inicia | Verificar `models/` antes de subir a API |
| Distribuição de entrada muito diferente do treino | Queda de AUC-ROC | Monitorar PSI semanal das features |
| Threshold desatualizado após mudança no negócio | Trade-off FP/FN fora de alvo | Revisar threshold mensalmente com equipe de negócio |
| Alta carga na API | Latência > 500ms | Monitorar p99, escalar horizontalmente |

---

## Plano de Monitoramento

| Métrica | Frequência | Alerta | Ação |
|---|---|---|---|
| AUC-ROC em amostra recente | Semanal | < 0.75 | Analisar dados, retreinar se necessário |
| PSI das features numéricas | Semanal | > 0.20 | Investigar data drift, acionar retraining |
| Taxa de churn real vs. previsto | Mensal | Desvio > 10 p.p. | Recalibrar threshold ou retreinar |
| Latência p99 da API | Contínua | > 500 ms | Escalar serviço, investigar gargalos |
| Taxa de erros HTTP 5xx | Contínua | > 1% | Investigar logs imediatamente |
| Taxa de requisições com 422 | Diária | > 5% | Revisar schema de entrada, comunicar clientes da API |
