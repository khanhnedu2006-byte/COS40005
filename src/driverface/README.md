# driverface – Phase 2 (not started)

Rules before touching any cabin video:
1. Written consent from every driver (Law 91/2025/QH15, Art. 9; biometric data = sensitive, Art. 31).
2. Process **offline** on an encrypted drive in Vietnam. No cloud (no AWS, no Google Colab – cross-border transfer, Art. 20).
3. Store features (EAR, MAR, head pose), not raw frames, wherever possible; delete raw video after the retention period.
4. Never commit video or frames to git.

Planned modules: `preprocess.py` (frame sampling, face crop), `features.py`, `model.py`, `infer.py`.
