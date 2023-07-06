import torch


def enable_dropout(model_module):
    for m in model_module.modules():
        if m._get_name() == "Dropout":
            m.train()


def get_monte_carlo_predictions(batch, model_module, n_ensemble_members):
    model_module.eval()
    enable_dropout(model_module)
    ensemble_predictions = []
    for _ in range(n_ensemble_members):
        with torch.no_grad():
            prediction = model_module.forward(batch)
        ensemble_predictions.append(prediction)
    ensemble_predictions = torch.stack(ensemble_predictions)
    return ensemble_predictions
