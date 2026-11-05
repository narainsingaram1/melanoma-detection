"""
Optimized master script - runs each phase with practical training configurations.
Designed for Apple Silicon MPS with reasonable training times.
"""
import os
import sys
import json
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.utils import set_seed, get_device, save_model, CLASSES, CLASS_NAMES
import torch


def run_phase1(data_dir, metadata_path):
    """Phase 1: Baseline CNN training."""
    from src.dataset import prepare_data, get_dataloaders, get_class_weights
    from src.baseline_cnn import SkinLesionCNN
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves, save_results, plot_class_distribution
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    device = get_device()
    print(f"\n{'='*60}")
    print(f"PHASE 1: BASELINE CNN TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")

    train_df, val_df, test_df = prepare_data(metadata_path)
    plot_class_distribution(train_df['label_idx'].tolist(), 'Training Set Class Distribution', 'results/class_distribution_train.png')

    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, data_dir, batch_size=32)
    class_weights = get_class_weights(train_df).to(device)

    model = SkinLesionCNN(num_classes=7).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(20):  # 20 epochs
        model.train()
        rl, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            rl += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        tl, ta = rl / total, correct / total

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                vl += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                vt += labels.size(0)
                vc += pred.eq(labels).sum().item()
        vl, va = vl / vt, vc / vt
        scheduler.step(vl)

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        status = ""
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            save_model(model, "results/best_baseline.pth")
            status = " [BEST]"
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"Epoch {epoch+1}: TL={tl:.4f} TA={ta:.4f} VL={vl:.4f} VA={va:.4f}{status}")
                print("Early stopping!")
                break
        print(f"Epoch {epoch+1}: TL={tl:.4f} TA={ta:.4f} VL={vl:.4f} VA={va:.4f}{status}")
        sys.stdout.flush()

    plot_training_curves(history, "results/baseline_training_curves.png")

    model.load_state_dict(torch.load("results/best_baseline.pth", map_location=device, weights_only=True))
    model.to(device)
    results = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(results['y_true'], results['y_pred'], 'results/baseline_confusion_matrix.png', 'Baseline CNN')
    save_results(results, 'results/baseline_metrics.json')

    mel_recall = results['report'].get('mel', {}).get('recall', 0)
    print(f"\nBaseline: Acc={results['accuracy']:.4f} F1={results['macro_f1']:.4f} MelRecall={mel_recall:.4f}")
    sys.stdout.flush()
    return results


def run_phase2(data_dir, metadata_path):
    """Phase 2: Transfer learning (EfficientNet-B0)."""
    from src.dataset import prepare_data, get_dataloaders, get_class_weights
    from src.transfer_model import SkinLesionTransfer
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves, save_results
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    device = get_device()
    print(f"\n{'='*60}")
    print(f"PHASE 2: TRANSFER LEARNING (EfficientNet-B0)")
    print(f"{'='*60}")

    train_df, val_df, test_df = prepare_data(metadata_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, data_dir, batch_size=32)
    class_weights = get_class_weights(train_df).to(device)

    model = SkinLesionTransfer(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    def train_epochs(optimizer, scheduler, n_epochs, phase_name):
        nonlocal best_val_loss
        patience_counter = 0
        for epoch in range(n_epochs):
            model.train()
            rl, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                rl += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
            tl, ta = rl / total, correct / total

            model.eval()
            vl, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    vl += loss.item() * inputs.size(0)
                    _, pred = outputs.max(1)
                    vt += labels.size(0)
                    vc += pred.eq(labels).sum().item()
            vl, va = vl / vt, vc / vt
            scheduler.step(vl)

            history['train_loss'].append(tl)
            history['val_loss'].append(vl)
            history['train_acc'].append(ta)
            history['val_acc'].append(va)

            status = ""
            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                save_model(model, "results/best_transfer.pth")
                status = " [BEST]"
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"[{phase_name}] Ep {len(history['train_loss'])}: TL={tl:.4f} TA={ta:.4f} VL={vl:.4f} VA={va:.4f}{status}")
                    print("Early stopping!")
                    break
            print(f"[{phase_name}] Ep {len(history['train_loss'])}: TL={tl:.4f} TA={ta:.4f} VL={vl:.4f} VA={va:.4f}{status}")
            sys.stdout.flush()

    # Stage 1: Head only
    print("--- Stage 1: Training classifier head ---")
    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    sch1 = ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=3)
    train_epochs(opt1, sch1, 8, "Head")

    # Stage 2: Full fine-tune
    print("\n--- Stage 2: Full fine-tuning ---")
    model.unfreeze_all()
    opt2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch2 = ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=3)
    train_epochs(opt2, sch2, 12, "Full")

    plot_training_curves(history, "results/transfer_training_curves.png")

    model.load_state_dict(torch.load("results/best_transfer.pth", map_location=device, weights_only=True))
    model.to(device)
    results = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(results['y_true'], results['y_pred'], 'results/transfer_confusion_matrix.png', 'Transfer (EfficientNet-B0)')
    save_results(results, 'results/transfer_metrics.json')

    mel_recall = results['report'].get('mel', {}).get('recall', 0)
    print(f"\nTransfer: Acc={results['accuracy']:.4f} F1={results['macro_f1']:.4f} MelRecall={mel_recall:.4f}")
    sys.stdout.flush()
    return results


def run_phase3(data_dir, metadata_path, baseline_results, transfer_results):
    """Phase 3: GAN augmentation + retrain."""
    from src.dataset import prepare_data, get_dataloaders, get_class_weights, HAM10000Dataset, get_transforms
    from src.dcgan import Generator, Discriminator
    from src.transfer_model import SkinLesionTransfer
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves, save_results, compare_models
    from torch.utils.data import DataLoader, Dataset, ConcatDataset
    from torchvision.utils import save_image
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np

    device = get_device()
    print(f"\n{'='*60}")
    print(f"PHASE 3: GAN-BASED DATA AUGMENTATION")
    print(f"{'='*60}")

    train_df, val_df, test_df = prepare_data(metadata_path)
    train_transform, eval_transform = get_transforms()

    train_dataset = HAM10000Dataset(train_df, data_dir, transform=train_transform)
    val_loader = DataLoader(HAM10000Dataset(val_df, data_dir, transform=eval_transform), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(HAM10000Dataset(test_df, data_dir, transform=eval_transform), batch_size=32, shuffle=False, num_workers=0)

    rare_indices = [CLASSES.index('mel'), CLASSES.index('df')]
    class_counts = train_df['label_idx'].value_counts().sort_index()
    majority_count = class_counts.max()

    print(f"Rare classes: mel (idx={rare_indices[0]}), df (idx={rare_indices[1]})")
    for idx, cls in enumerate(CLASSES):
        print(f"  {cls}: {class_counts.get(idx, 0)}")

    # Filter rare class samples
    print("\nCollecting rare class samples...")
    rare_samples = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        if label in rare_indices:
            rare_samples.append((img, label))
        if i % 1000 == 0:
            print(f"  Scanned {i}/{len(train_dataset)}...")
            sys.stdout.flush()
    print(f"  Found {len(rare_samples)} rare class samples")

    class SimpleDataset(Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx): return self.samples[idx]

    class SyntheticDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        def __len__(self): return len(self.images)
        def __getitem__(self, idx): return self.images[idx], self.labels[idx]

    rare_dataset = SimpleDataset(rare_samples)
    bs = min(16, len(rare_dataset))
    gan_loader = DataLoader(rare_dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)

    # Train GAN
    print("\n--- Training Conditional DCGAN ---")
    z_dim = 100
    G = Generator(z_dim=z_dim, num_classes=7).to(device)
    D = Discriminator(num_classes=7, img_size=224).to(device)
    G.weights_init()
    D.weights_init()

    bce = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    os.makedirs('results/gan_progress', exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g_loss_hist, d_loss_hist = [], []

    for epoch in range(30):  # 30 GAN epochs
        G.train(); D.train()
        dl, gl = [], []

        for real_imgs, labels in gan_loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs_cur = real_imgs.size(0)
            valid = torch.ones(bs_cur, 1, device=device) * 0.9
            fake_t = torch.zeros(bs_cur, 1, device=device) + 0.1

            # Discriminator
            opt_D.zero_grad()
            d_real = bce(D(real_imgs, labels), valid)
            z = torch.randn(bs_cur, z_dim, device=device)
            gen = G(z, labels)
            d_fake = bce(D(gen.detach(), labels), fake_t)
            d_loss = (d_real + d_fake) / 2
            d_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            z = torch.randn(bs_cur, z_dim, device=device)
            gen = G(z, labels)
            g_loss = bce(D(gen, labels), torch.ones(bs_cur, 1, device=device))
            g_loss.backward()
            opt_G.step()

            dl.append(d_loss.item()); gl.append(g_loss.item())

        avg_d, avg_g = np.mean(dl), np.mean(gl)
        d_loss_hist.append(avg_d); g_loss_hist.append(avg_g)

        if (epoch+1) % 5 == 0:
            print(f"  GAN Epoch {epoch+1}/30: D={avg_d:.4f} G={avg_g:.4f}")
            sys.stdout.flush()

        if (epoch+1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                tz = torch.randn(8, z_dim, device=device)
                tl_labels = torch.tensor([rare_indices[0]]*4 + [rare_indices[1]]*4, dtype=torch.long, device=device)
                imgs = G(tz, tl_labels)
                save_image(imgs, f"results/gan_progress/epoch_{epoch+1}.png", nrow=4, normalize=True)

    # Save GAN
    torch.save(G.state_dict(), 'results/generator.pth')

    # Plot GAN losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_loss_hist, label='Discriminator'); ax.plot(g_loss_hist, label='Generator')
    ax.set_title('GAN Training Losses'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig('results/gan_loss_curves.png', dpi=150); plt.close()

    # Generate synthetic data
    print("\n--- Generating Synthetic Images ---")
    G.eval()
    synthetic_images, synthetic_labels = [], []
    with torch.no_grad():
        for cls_idx in rare_indices:
            needed = min(1500, majority_count - class_counts.get(cls_idx, 0))
            needed = max(needed, 500)
            print(f"  Generating {needed} for {CLASSES[cls_idx]}...")
            generated = 0
            while generated < needed:
                bsz = min(32, needed - generated)
                z = torch.randn(bsz, z_dim, device=device)
                lbl = torch.full((bsz,), cls_idx, dtype=torch.long, device=device)
                gen = G(z, lbl).cpu()
                synthetic_images.extend(list(gen))
                synthetic_labels.extend([cls_idx] * bsz)
                generated += bsz
            sys.stdout.flush()

    print(f"  Total synthetic: {len(synthetic_images)}")

    # Retrain classifier
    print("\n--- Retraining Classifier on Augmented Data ---")
    syn_dataset = SyntheticDataset(synthetic_images, synthetic_labels)
    aug_dataset = ConcatDataset([train_dataset, syn_dataset])
    aug_loader = DataLoader(aug_dataset, batch_size=32, shuffle=True, num_workers=0)
    print(f"  Original: {len(train_dataset)}, Synthetic: {len(syn_dataset)}, Total: {len(aug_dataset)}")

    model = SkinLesionTransfer(num_classes=7).to(device)
    model.unfreeze_all()
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_vl = float('inf')

    for epoch in range(15):
        model.train()
        rl, correct, total = 0.0, 0, 0
        for inputs, labels in aug_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            rl += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        tl, ta = rl / total, correct / total

        model.eval()
        vl_sum, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                vl_sum += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                vt += labels.size(0)
                vc += pred.eq(labels).sum().item()
        vl, va = vl_sum / vt, vc / vt
        scheduler.step(vl)

        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta); history['val_acc'].append(va)

        status = ""
        if vl < best_vl:
            best_vl = vl
            save_model(model, 'results/best_gan_augmented.pth')
            status = " [BEST]"
        print(f"  [Aug] Ep {epoch+1}: TL={tl:.4f} TA={ta:.4f} VL={vl:.4f} VA={va:.4f}{status}")
        sys.stdout.flush()

    plot_training_curves(history, 'results/gan_augmented_training_curves.png')

    model.load_state_dict(torch.load('results/best_gan_augmented.pth', map_location=device, weights_only=True))
    model.to(device)
    gan_results = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(gan_results['y_true'], gan_results['y_pred'], 'results/gan_augmented_confusion_matrix.png', 'GAN-Augmented Transfer')
    save_results(gan_results, 'results/gan_augmented_metrics.json')

    # Compare all models
    if baseline_results and transfer_results:
        all_res = {
            'Baseline CNN': baseline_results,
            'Transfer (EfficientNet)': transfer_results,
            'Transfer + GAN Aug': gan_results
        }
        compare_models(all_res, 'results/model_comparison.png')
        print(f"\n{'Model':<30} {'Accuracy':<10} {'F1':<10} {'MelRecall':<10}")
        print("-" * 60)
        for n, r in all_res.items():
            mr = r['report'].get('mel', {}).get('recall', 0)
            print(f"{n:<30} {r['accuracy']:<10.4f} {r['macro_f1']:<10.4f} {mr:<10.4f}")

    return gan_results


def run_phase4(data_dir, metadata_path):
    """Phase 4: Grad-CAM explainability."""
    from src.dataset import prepare_data, get_dataloaders
    from src.transfer_model import SkinLesionTransfer
    from src.baseline_cnn import SkinLesionCNN
    from src.gradcam import generate_gradcam_grid

    device = get_device()
    print(f"\n{'='*60}")
    print(f"PHASE 4: GRAD-CAM EXPLAINABILITY")
    print(f"{'='*60}")

    train_df, val_df, test_df = prepare_data(metadata_path)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df, data_dir)

    models_to_visualize = [
        ('results/best_gan_augmented.pth', SkinLesionTransfer, 'transfer', 'results/gradcam_gan_aug'),
        ('results/best_transfer.pth', SkinLesionTransfer, 'transfer', 'results/gradcam_transfer'),
        ('results/best_baseline.pth', SkinLesionCNN, 'baseline', 'results/gradcam_baseline'),
    ]

    for model_path, ModelClass, model_type, save_dir in models_to_visualize:
        if os.path.exists(model_path):
            print(f"\nGrad-CAM for {model_path}...")
            model = ModelClass(num_classes=7).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            generate_gradcam_grid(model, test_loader, device, model_type=model_type,
                                  num_images=2, save_dir=save_dir)
            sys.stdout.flush()


def main():
    set_seed(42)
    device = get_device()
    start = time.time()

    print(f"{'='*60}")
    print(f"MELANOMA DETECTION WITH GAN-BASED CLASS BALANCING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    sys.stdout.flush()

    data_dir = os.path.join(project_root, 'data', 'ham10000')
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        print("Downloading dataset...")
        import kagglehub
        data_dir = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
        metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')

    os.makedirs('results', exist_ok=True)

    # Phase 1
    t1 = time.time()
    if os.path.exists('results/best_baseline.pth') and os.path.exists('results/baseline_metrics.json'):
        print("\nPhase 1 already complete! Loading baseline results from results/baseline_metrics.json...")
        with open('results/baseline_metrics.json') as f:
            bm = json.load(f)
        baseline_results = {
            'accuracy': bm['accuracy'],
            'macro_f1': bm['macro_f1'],
            'report': bm['per_class']
        }
        print(f"Loaded Baseline: Acc={bm['accuracy']:.4f} F1={bm['macro_f1']:.4f} MelRecall={bm['per_class']['mel']['recall']:.4f}")
    else:
        baseline_results = run_phase1(data_dir, metadata_path)
    print(f"Phase 1 time: {time.time()-t1:.0f}s")

    # Phase 2
    t2 = time.time()
    transfer_results = run_phase2(data_dir, metadata_path)
    print(f"Phase 2 time: {time.time()-t2:.0f}s")

    # Phase 3
    t3 = time.time()
    gan_results = run_phase3(data_dir, metadata_path, baseline_results, transfer_results)
    print(f"Phase 3 time: {time.time()-t3:.0f}s")

    # Phase 4
    t4 = time.time()
    run_phase4(data_dir, metadata_path)
    print(f"Phase 4 time: {time.time()-t4:.0f}s")

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE! Total: {total:.0f}s ({total/60:.1f}min)")
    print(f"{'='*60}")
    print(f"\nTo launch the Gradio demo: python app.py")

    # Save summary
    summary = {'total_time': total, 'device': str(device), 'models': {}}
    for name, res in [('Baseline', baseline_results), ('Transfer', transfer_results), ('GAN-Aug', gan_results)]:
        if res:
            summary['models'][name] = {
                'accuracy': res['accuracy'],
                'macro_f1': res['macro_f1'],
                'mel_recall': res['report'].get('mel', {}).get('recall', 0),
                'df_recall': res['report'].get('df', {}).get('recall', 0),
            }
    with open('results/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to results/final_summary.json")


if __name__ == "__main__":
    main()
