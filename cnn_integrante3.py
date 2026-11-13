"""Integrante 3: CNN ResNet18, treinamento, avaliação e Grad-CAM.
Uso: python cnn_integrante3.py --dados segunda_leitura_processado.zip --versao realcado
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ['normal', 'pneumonia', 'tuberculose']
SPLITS = {'treino': 'treino', 'validacao': 'validacao', 'teste': 'teste'}

class Radiografias(Dataset):
    def __init__(self, fonte, versao, split, transform):
        self.fonte = Path(fonte)
        self.versao, self.split, self.transform = versao, split, transform
        self.itens = []
        if self.fonte.is_file() and self.fonte.suffix.lower() == '.zip':
            with zipfile.ZipFile(self.fonte) as z:
                nomes = set(z.namelist())
            for i, classe in enumerate(CLASSES):
                prefixo = f'processado/{versao}/{split}/{classe}/'
                self.itens += [(n, i) for n in sorted(nomes) if n.startswith(prefixo) and n.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            for i, classe in enumerate(CLASSES):
                pasta = self.fonte / 'processado' / versao / split / classe
                self.itens += [(str(p), i) for p in sorted(pasta.glob('*')) if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        if not self.itens:
            raise ValueError(f'Nenhuma imagem encontrada: {self.fonte}, {versao}, {split}')
        self.z = None

    def __len__(self): return len(self.itens)

    def __getitem__(self, idx):
        caminho, classe = self.itens[idx]
        if self.fonte.is_file():
            import io
            if self.z is None: self.z = zipfile.ZipFile(self.fonte)
            with self.z.open(caminho) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
        else:
            with Image.open(caminho) as f: img = f.convert('RGB')
        return self.transform(img), classe, caminho


def criar_transformacoes(tamanho):
    normalizar = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    treino = transforms.Compose([transforms.Resize((tamanho, tamanho)),
        transforms.RandomRotation(7), transforms.RandomAffine(0, translate=(0.03, 0.03)),
        transforms.ToTensor(), normalizar])
    avaliacao = transforms.Compose([transforms.Resize((tamanho, tamanho)), transforms.ToTensor(), normalizar])
    return treino, avaliacao


def criar_modelo(pretreinado=True):
    pesos = models.ResNet18_Weights.DEFAULT if pretreinado else None
    modelo = models.resnet18(weights=pesos)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASSES))
    return modelo


def executar_epoca(modelo, loader, criterio, dispositivo, otimizador=None):
    treinando = otimizador is not None
    modelo.train(treinando)
    perdas, y, pred = [], [], []
    for imagens, rotulos, _ in loader:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
        with torch.set_grad_enabled(treinando):
            logits = modelo(imagens)
            perda = criterio(logits, rotulos)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
        perdas.append((perda.item(), len(rotulos)))
        y.extend(rotulos.cpu().tolist())
        pred.extend(logits.argmax(1).detach().cpu().tolist())
    return sum(p * n for p, n in perdas) / sum(n for _, n in perdas), accuracy_score(y, pred), f1_score(y, pred, average='macro', zero_division=0)


def avaliar(modelo, loader, dispositivo, pasta):
    modelo.eval()
    verdade, previsoes, probabilidades, caminhos = [], [], [], []
    with torch.no_grad():
        for imagens, rotulos, nomes in loader:
            logits = modelo(imagens.to(dispositivo))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            verdade.extend(rotulos.tolist())
            previsoes.extend(probs.argmax(axis=1).tolist())
            probabilidades.extend(probs.tolist())
            caminhos.extend(nomes)
    verdade, previsoes, probabilidades = np.array(verdade), np.array(previsoes), np.array(probabilidades)
    relatorio = classification_report(verdade, previsoes, labels=list(range(3)), target_names=CLASSES, output_dict=True, zero_division=0)
    metricas = {'acuracia': float(accuracy_score(verdade, previsoes)),
                'f1_macro': float(f1_score(verdade, previsoes, average='macro', zero_division=0)),
                'relatorio_por_classe': relatorio}
    try:
        metricas['auc_roc_ovr_macro'] = float(roc_auc_score(verdade, probabilidades, multi_class='ovr', average='macro', labels=[0,1,2]))
    except ValueError:
        metricas['auc_roc_ovr_macro'] = None
    for i, nome in enumerate(CLASSES):
        tp = int(((verdade == i) & (previsoes == i)).sum())
        tn = int(((verdade != i) & (previsoes != i)).sum())
        fp = int(((verdade != i) & (previsoes == i)).sum())
        fn = int(((verdade == i) & (previsoes != i)).sum())
        metricas[f'especificidade_{nome}'] = tn / (tn + fp) if tn + fp else None
        metricas[f'sensibilidade_{nome}'] = tp / (tp + fn) if tp + fn else None
    (pasta / 'metricas_teste.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf8')
    with (pasta / 'predicoes_teste.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['arquivo', 'classe_real', 'classe_predita'] + [f'prob_{c}' for c in CLASSES])
        for nome, real, pred, prob in zip(caminhos, verdade, previsoes, probabilidades):
            w.writerow([nome, CLASSES[real], CLASSES[pred], *prob])
    matriz = confusion_matrix(verdade, previsoes, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matriz, cmap='Blues')
    for (i,j), v in np.ndenumerate(matriz): ax.text(j, i, str(v), ha='center', va='center')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predição', ylabel='Classe real', title='Matriz de confusão - teste')
    fig.tight_layout(); fig.savefig(pasta / 'matriz_confusao.png', dpi=160); plt.close(fig)
    return metricas


def gradcam(modelo, tensor, classe, dispositivo):
    """Grad-CAM sobre a última camada convolucional da ResNet18."""
    ativacoes, gradientes = [], []
    def forward_hook(_mod, _entrada, saida):
        ativacoes.append(saida.detach())
        saida.register_hook(lambda g: gradientes.append(g.detach()))
    handle = modelo.layer4[-1].conv2.register_forward_hook(forward_hook)
    try:
        modelo.eval()
        with torch.enable_grad():
            modelo.zero_grad(set_to_none=True)
            saida = modelo(tensor.unsqueeze(0).to(dispositivo))
            saida[0, classe].backward()
        pesos = gradientes[0].mean(dim=(2,3), keepdim=True)
        mapa = torch.relu((pesos * ativacoes[0]).sum(dim=1))[0].cpu().numpy()
        mapa -= mapa.min()
        mapa /= mapa.max() + 1e-8
        return mapa
    finally:
        handle.remove()


def salvar_gradcams(modelo, dataset, dispositivo, pasta, quantidade=9):
    pasta.mkdir(parents=True, exist_ok=True)
    por_classe = {i: [] for i in range(3)}
    for idx, (_, classe) in enumerate(dataset.itens): por_classe[classe].append(idx)
    indices = []
    for i in range(3): indices.extend(por_classe[i][:max(1, quantidade // 3)])
    for idx in indices[:quantidade]:
        tensor, real, caminho = dataset[idx]
        with torch.no_grad():
            prob = torch.softmax(modelo(tensor.unsqueeze(0).to(dispositivo)), dim=1)[0]
            pred = int(prob.argmax())
            confianca = float(prob[pred])
        mapa = gradcam(modelo, tensor, pred, dispositivo)
        original = tensor.numpy().transpose(1,2,0)
        original = np.clip(original * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0, 1)
        from PIL import Image as PILImage
        mapa = np.asarray(PILImage.fromarray(np.uint8(mapa*255)).resize((original.shape[1],original.shape[0]), PILImage.Resampling.BILINEAR)) / 255.0
        fig, eixos = plt.subplots(1, 2, figsize=(9,4))
        eixos[0].imshow(original); eixos[0].set_title(f'Real: {CLASSES[real]}')
        eixos[1].imshow(original); eixos[1].imshow(mapa, cmap='jet', alpha=0.40, vmin=0, vmax=1)
        eixos[1].set_title(f'Grad-CAM: {CLASSES[pred]} ({confianca:.1%})')
        for ax in eixos: ax.axis('off')
        fig.tight_layout()
        fig.savefig(pasta / f'{idx:04d}_{Path(caminho).stem}.png', dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='CNN para raio-X: normal, pneumonia, tuberculose')
    parser.add_argument('--dados', default='segunda_leitura_processado.zip')
    parser.add_argument('--versao', choices=['basico', 'realcado'], default='realcado')
    parser.add_argument('--saida', default=None)
    parser.add_argument('--epocas', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--paciencia', type=int, default=5)
    parser.add_argument('--tamanho', type=int, default=224)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--sem-pesos', action='store_true', help='Não usar pesos ImageNet (apenas para teste técnico)')
    args = parser.parse_args()
    if args.epocas < 1 or args.batch < 1 or args.paciencia < 1: parser.error('epocas, batch e paciencia devem ser positivos')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    pasta = Path(args.saida or f'resultados_cnn_{args.versao}')
    pasta.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_treino, t_avaliacao = criar_transformacoes(args.tamanho)
    datasets = {s: Radiografias(args.dados, args.versao, s, t_treino if s == 'treino' else t_avaliacao) for s in SPLITS}
    for s, ds in datasets.items():
        contagens = np.bincount([classe for _, classe in ds.itens], minlength=3)
        print(s, dict(zip(CLASSES, contagens.tolist())))
    loaders = {s: DataLoader(ds, batch_size=args.batch, shuffle=(s=='treino'), num_workers=args.workers) for s, ds in datasets.items()}
    modelo = criar_modelo(not args.sem_pesos).to(dispositivo)
    for parametro in modelo.parameters(): parametro.requires_grad = False
    for parametro in modelo.fc.parameters(): parametro.requires_grad = True
    contagens = np.bincount([classe for _, classe in datasets['treino'].itens], minlength=3)
    pesos = torch.tensor(len(datasets['treino']) / (3 * contagens), dtype=torch.float32, device=dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    otimizador = torch.optim.AdamW(modelo.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    historico, melhor, espera = [], -1.0, 0
    for epoca in range(1, args.epocas+1):
        # Ajuste fino das últimas camadas após aquecimento inicial
        if epoca == 6:
            for parametro in modelo.layer4.parameters(): parametro.requires_grad = True
            otimizador = torch.optim.AdamW([
                {'params': modelo.layer4.parameters(), 'lr': args.lr * 0.1},
                {'params': modelo.fc.parameters(), 'lr': args.lr * 0.3}], weight_decay=1e-4)
        treino = executar_epoca(modelo, loaders['treino'], criterio, dispositivo, otimizador)
        validacao = executar_epoca(modelo, loaders['validacao'], criterio, dispositivo)
        historico.append({'epoca': epoca, 'loss_treino': treino[0], 'acc_treino': treino[1],
                          'f1_treino': treino[2], 'loss_validacao': validacao[0],
                          'acc_validacao': validacao[1], 'f1_validacao': validacao[2]})
        print(f'Epoca {epoca:02d} | loss treino {treino[0]:.4f} | loss val {validacao[0]:.4f} | F1 val {validacao[2]:.4f}')
        if validacao[2] > melhor + 1e-5:
            melhor, espera = validacao[2], 0
            torch.save({'state_dict': modelo.state_dict(), 'classes': CLASSES,
                        'tamanho': args.tamanho, 'versao': args.versao}, pasta / 'melhor_modelo.pt')
        else:
            espera += 1
            if espera >= args.paciencia:
                print('Parada antecipada.'); break
    with (pasta / 'historico.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(historico[0])); w.writeheader(); w.writerows(historico)
    fig, axs = plt.subplots(1, 2, figsize=(11,4))
    for ax, campo, titulo in [(axs[0], 'loss', 'Perda'), (axs[1], 'f1', 'F1 macro')]:
        for split in ('treino', 'validacao'):
            ax.plot([h['epoca'] for h in historico], [h[f'{campo}_{split}'] for h in historico], label=split)
        ax.set(title=titulo, xlabel='Época'); ax.legend()
    fig.tight_layout(); fig.savefig(pasta / 'curvas_treinamento.png', dpi=160); plt.close(fig)
    checkpoint = torch.load(pasta / 'melhor_modelo.pt', map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(checkpoint['state_dict'])
    metricas = avaliar(modelo, loaders['teste'], dispositivo, pasta)
    salvar_gradcams(modelo, datasets['teste'], dispositivo, pasta / 'gradcam')
    print('Resultados salvos em', pasta.resolve())
    print('Acuracia teste:', round(metricas['acuracia'], 4), '| F1 macro:', round(metricas['f1_macro'], 4))

if __name__ == '__main__': main()
