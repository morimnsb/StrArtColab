import glob, cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from importance.build_map import build_importance_map
from ml.pattern_model import PatchCNN

class PatchDataset(Dataset):
    def __init__(self, paths, size=(400,400), patch=32, samples=3000):
        self.items=[]; self.patch=patch; ph=patch//2
        for p in paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, size)
            M,_ = build_importance_map(g)
            H,W = g.shape
            ys = np.random.randint(ph, H-ph, size=samples)
            xs = np.random.randint(ph, W-ph, size=samples)
            for x,y in zip(xs,ys):
                patch_img = g[y-ph:y+ph, x-ph:x+ph].astype(np.float32)/255.0
                target = M[y-4:y+5, x-4:x+5].mean().astype(np.float32)
                self.items.append((patch_img, target))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img,t = self.items[i]
        return torch.from_numpy(img)[None,...], torch.tensor(t, dtype=torch.float32)

def main():
    paths = glob.glob("data/train_images/*.*")
    ds = PatchDataset(paths, samples=3000)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()

    for ep in range(5):
        total=0
        for x,y in tqdm(dl, desc=f"epoch {ep+1}"):
            x=x.to(device); y=y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = lossf(pred, y)
            loss.backward(); opt.step()
            total += loss.item()*x.size(0)
        print(f"epoch {ep+1} mean loss: {total/len(ds):.6f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pattern_cnn.pth")
    print("✅ saved checkpoints/pattern_cnn.pth")
if __name__=="__main__": main()
