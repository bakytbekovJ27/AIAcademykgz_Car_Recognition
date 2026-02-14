import os
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.core import CNN, CNNTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Используется устройство: {device}")


class CarBrandTrainer:
    def __init__(self, dataset_path='dataset', model_type='CNN', batch_size=32, 
                 num_epochs=50, learning_rate=0.001):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = nn.NLLLoss()
        self.optimizer = None
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self):
        """Подготовка данных для обучения"""
        train_path = os.path.join(self.dataset_path, 'train')
        val_path = os.path.join(self.dataset_path, 'val')
        test_path = os.path.join(self.dataset_path, 'test')
        
        # Проверка существования папок
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Папка {train_path} не найдена!")
        
        # Определение трансформаций
        if self.model_type == 'CNN':
            img_size = (96, 128)
        else:  # CNNTransfer
            img_size = (384, 384)
            
        train_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Загрузка train dataset
        full_train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
        
        # Создание маппинга классов
        class_to_id = {cls_name: idx for idx, cls_name in enumerate(full_train_dataset.classes)}
        id_to_class = {str(idx): cls_name for cls_name, idx in class_to_id.items()}
        
        # Сохранение маппинга
        with open('class_id.json', 'w') as f:
            json.dump({'class_to_id': class_to_id, 'id_to_class': id_to_class}, f, indent=4)
        
        print(f"Найдено классов: {len(full_train_dataset.classes)}")
        print(f"Классы: {full_train_dataset.classes}")
        
        # Если нет валидационного набора, создаем из train (10%)
        if not os.path.exists(val_path) or len(os.listdir(val_path)) == 0:
            print("Валидационный набор не найден. Создание из train (10%)...")
            train_size = int(0.9 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        else:
            train_dataset = full_train_dataset
            val_dataset = datasets.ImageFolder(val_path, transform=test_transforms)
        
        # Загрузка test dataset
        if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
            test_dataset = datasets.ImageFolder(test_path, transform=test_transforms)
        else:
            print("Тестовый набор не найден. Используется валидационный набор для тестирования.")
            test_dataset = val_dataset
        
        # Создание DataLoader'ов
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                      shuffle=False, num_workers=4)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return len(full_train_dataset.classes)
        
    def build_model(self, num_classes):
        """Создание модели"""
        if self.model_type == 'CNN':
            self.model = CNN(classes=num_classes)
            print("Создана модель CNN")
        else:
            self.model = CNNTransfer(classes=num_classes)
            print("Создана модель CNNTransfer (EfficientNet)")
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train_epoch(self):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 
                            'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
        
    def validate(self):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
        
    def test(self):
        """Тестирование модели"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        return test_acc
        
    def train(self):
        """Полный цикл обучения"""
        print(f"\nНачало обучения модели {self.model_type}")
        print("=" * 50)
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nЭпоха {epoch + 1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best')
                print(f"Сохранена лучшая модель с точностью {val_acc:.2f}%")
        
        # Сохранение финальной модели
        self.save_model('final')
        
        # Тестирование
        print("\nТестирование модели...")
        test_acc = self.test()
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Построение графиков
        self.plot_training_history()
        
    def save_model(self, suffix=''):
        """Сохранение модели"""
        os.makedirs('models', exist_ok=True)
        
        if self.model_type == 'CNN':
            filename = f'models/model_ft_{suffix}.pt' if suffix else 'models/model_ft_v1.pt'
        else:
            filename = f'models/model_tl_{suffix}.pt' if suffix else 'models/model_tl_v1.pt'
        
        torch.save(self.model.state_dict(), filename)
        print(f"Модель сохранена: {filename}")
        
    def plot_training_history(self):
        """Построение графиков обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(self.train_accuracies, label='Train Accuracy', marker='o')
        ax2.plot(self.val_accuracies, label='Val Accuracy', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_type}.png')
        print(f"График сохранен: training_history_{self.model_type}.png")
        plt.close()


def main():
    print("=" * 70)
    print("Car Brand Recognition - Model Training")
    print("=" * 70)
    
    # Параметры обучения
    DATASET_PATH = 'dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Выбор модели для обучения
    print("\nВыберите модель для обучения:")
    print("1. CNN (быстрая, легкая модель)")
    print("2. CNNTransfer (EfficientNet, более точная)")
    print("3. Обе модели")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    models_to_train = []
    if choice == '1':
        models_to_train = ['CNN']
    elif choice == '2':
        models_to_train = ['CNNTransfer']
    elif choice == '3':
        models_to_train = ['CNN', 'CNNTransfer']
    else:
        print("Неверный выбор!")
        return
    
    # Обучение выбранных моделей
    for model_type in models_to_train:
        print("\n" + "=" * 70)
        trainer = CarBrandTrainer(
            dataset_path=DATASET_PATH,
            model_type=model_type,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE
        )
        
        try:
            num_classes = trainer.prepare_data()
            trainer.build_model(num_classes)
            trainer.train()
        except Exception as e:
            print(f"Ошибка при обучении модели {model_type}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Обучение завершено!")
    print("=" * 70)


if __name__ == "__main__":
    main()