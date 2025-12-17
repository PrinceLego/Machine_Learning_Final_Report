import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report  # ✨ 新增 classification_report

current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = "experiments"
RUN_DIR = os.path.join(base_dir, current_timestamp)

if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

BEST_MODEL_PATH = os.path.join(RUN_DIR, 'best_wafer_SACNN.keras')
FINAL_MODEL_PATH = os.path.join(RUN_DIR, 'final_wafer_SACNN.keras')
LOG_FILE_PATH = os.path.join(RUN_DIR, 'execution_result.txt')
HISTORY_IMG_PATH = os.path.join(RUN_DIR, 'training_history.png')
CM_IMG_PATH = os.path.join(RUN_DIR, 'confusion_matrix.png')
TENSORBOARD_LOG_DIR = os.path.join(RUN_DIR, 'logs')


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 已啟用：", gpus)
else:
    print(" 未偵測到 GPU，將使用 CPU")

TRAIN_FILE_PATH = 'synthetic_mixed_dataset_noise.npz'
TEST_FILE_PATH = 'multi_defect_wafers.npz'
try:
    print(f"正在載入訓練資料... {TRAIN_FILE_PATH}")
    train_data = np.load(TRAIN_FILE_PATH)
    X_train_full = train_data['arr_0']
    y_train_full = train_data['arr_1'] 
    
    print(f"正在載入測試資料... {TEST_FILE_PATH}")
    test_data = np.load(TEST_FILE_PATH)
    X_test = test_data['arr_0']
    y_test = test_data['arr_1']
    
except FileNotFoundError as e:
    print(f"錯誤：找不到 .npz 檔案。請檢查路徑是否正確。 {e}")
    exit() 
except KeyError as e:
    print(f"錯誤：.npz 檔案中找不到 'arr_0' 或 'arr_1'。請檢查檔案內容。 {e}")
    exit()

X_train_full = X_train_full.astype('float32') / 2.0
X_train_full = np.expand_dims(X_train_full, axis=-1)
X_test = X_test.astype('float32') / 2.0
X_test = np.expand_dims(X_test, axis=-1)

DEFECT_NAMES = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'S', 'R']
CLASS_TUPLES = [
    (0, 0, 0, 0, 1, 0, 1, 0), (0, 0, 0, 1, 0, 0, 1, 0), (0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 1, 0), (0, 0, 1, 0, 0, 0, 1, 0), (0, 0, 1, 0, 1, 0, 0, 0),
    (0, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 0, 0, 0, 1, 0), (0, 1, 0, 0, 1, 0, 0, 0),
    (0, 1, 0, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 1, 0),
    (0, 1, 0, 1, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0, 1, 0), (0, 1, 1, 0, 0, 0, 0, 0),
    (0, 1, 1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0, 1, 0),
    (1, 0, 0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 1, 0, 0, 0), (1, 0, 0, 0, 1, 0, 1, 0),
    (1, 0, 0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 1, 1, 0, 0, 0),
    (1, 0, 0, 1, 1, 0, 1, 0), (1, 0, 1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 1, 0),
    (1, 0, 1, 0, 1, 0, 0, 0), (1, 0, 1, 0, 1, 0, 1, 0)
]
def tuple_to_string(t):
    names = [DEFECT_NAMES[i] for i, v in enumerate(t) if v == 1]
    return "+".join(names)
CLASS_NAMES = [tuple_to_string(t) for t in CLASS_TUPLES]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {tuple(t): i for i, t in enumerate(CLASS_TUPLES)}

def convert_labels_to_int(y_batch_multi_label):
    y_batch_int = []
    for y_vec in y_batch_multi_label:
        y_tuple = tuple(y_vec)
        if y_tuple in LABEL_MAP:
            y_batch_int.append(LABEL_MAP[y_tuple])
        else:
            y_batch_int.append(-1)
    return np.array(y_batch_int)

print("\n正在將 (N, 8) 標籤轉換為 (N, 1) 整數標籤...")
y_train_full_int = convert_labels_to_int(y_train_full)
y_test_int = convert_labels_to_int(y_test)

train_mask = y_train_full_int != -1
X_train_full = X_train_full[train_mask]
y_train_full_int = y_train_full_int[train_mask]

test_mask = y_test_int != -1
X_test = X_test[test_mask]
y_test_int = y_test_int[test_mask]

# === 4.5 分割 訓練 / 驗證集 ===
X_train, X_val, y_train_int, y_val_int = train_test_split(
    X_train_full, y_train_full_int, 
    test_size=0.2, 
    random_state=42,
    stratify=y_train_full_int 
)
print(f"訓練集大小 (Train):   {X_train.shape}")
print(f"驗證集大小 (Val):     {X_val.shape}")
print(f"測試集大小 (Test):    {X_test.shape}")



def self_attention_layer(x, block_id):
    attention_scores = tf.keras.layers.Conv2D(1, (1, 1), activation='linear', name=f'sa_conv_block{block_id}')(x)
    attention_weights = tf.keras.layers.Softmax(axis=(1, 2), name=f'sa_softmax_block{block_id}')(attention_scores)
    scaled_features = tf.keras.layers.multiply([x, attention_weights], name=f'sa_multiply_block{block_id}')
    return scaled_features

def build_sacnn_model(input_shape=(52, 52, 1), num_classes=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=input_shape)


    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x) 
    x = tf.keras.layers.GaussianNoise(0.01)(x) 
    

    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)


    x = self_attention_layer(x, block_id=1) 


    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

 
    x = self_attention_layer(x, block_id=2)


    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_sacnn_model(input_shape=(52, 52, 1), num_classes=NUM_CLASSES)
model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    tensorboard_cb,
    reduce_lr_cb, 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True), 
    tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min') 
]


start_time = time.time()
history = model.fit(
    X_train, y_train_int, 
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val_int),
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
print(f"訓練總時間：{end_time - start_time:.2f} 秒")


model.save(FINAL_MODEL_PATH)
print(f"最終模型已儲存至: {FINAL_MODEL_PATH}")


print("\n正在繪製訓練歷史和混淆矩陣...")
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(HISTORY_IMG_PATH)
print(f"訓練歷史圖已儲存至: {HISTORY_IMG_PATH}")

try:
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
except Exception as e:
    print(f"載入最佳模型失敗: {e}，將使用當前模型")
    best_model = model

y_pred_probs = best_model.predict(X_test)
y_pred_int = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_int, y_pred_int)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix - Test Set\nRun: {current_timestamp}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(CM_IMG_PATH)
print(f"混淆矩陣已儲存至: {CM_IMG_PATH}")


print("\n" + "="*30)
print(" 正在載入「最佳 SACNN 模型」進行最終評估... ")
print("="*30)
test_results = best_model.evaluate(X_test, y_test_int, verbose=1)

print("\n正在生成詳細分類指標 (Table 6 格式)...")

report_dict = classification_report(y_test_int, y_pred_int, target_names=CLASS_NAMES, output_dict=True)

def format_table_row(class_name, p, r, f1, s):

    return f"{class_name:<20} {p:>10.2f} {r:>10.2f} {f1:>10.2f} {s:>10,}"

table_lines = []
table_lines.append("="*66)
table_lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
table_lines.append("-" * 66)


for class_name in CLASS_NAMES:
    if class_name in report_dict:
        res = report_dict[class_name]
        row_str = format_table_row(
            class_name, 
            res['precision'], 
            res['recall'], 
            res['f1-score'], 
            int(res['support'])
        )
        table_lines.append(row_str)

table_lines.append("-" * 66)


accuracy = report_dict['accuracy']
total_support = int(report_dict['macro avg']['support'])
table_lines.append(f"{'Accuracy':<20} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10,}")

macro = report_dict['macro avg']
table_lines.append(format_table_row("Macro Avg", macro['precision'], macro['recall'], macro['f1-score'], int(macro['support'])))

weighted = report_dict['weighted avg']
table_lines.append(format_table_row("Weighted Avg", weighted['precision'], weighted['recall'], weighted['f1-score'], int(weighted['support'])))

table_lines.append("="*66)

final_table_str = "\n".join(table_lines)

print("\n" + final_table_str)


log_content = (
    f"[{current_timestamp}] 實驗執行報告\n"
    f"{'='*66}\n"
    f"  - 實驗目錄:   {RUN_DIR}\n"
    f"  - 資料集:     {TEST_FILE_PATH}\n"
    f"  - 模型架構:   SACNN (29 Classes)\n"
    f"  - Epochs:     {len(history.history['loss'])}\n"
    f"  - 訓練總時間: {end_time - start_time:.2f} 秒\n"
    f"{'-'*30}\n"
    f"  ★ Global Test Loss:     {test_results[0]:.4f}\n"
    f"  ★ Global Test Accuracy: {test_results[1]:.4f}\n"
    f"\nDetailed Classification Report:\n"
    f"{final_table_str}\n"
)

try:
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"\n完整測試報告 (含 Table 6) 已寫入： {LOG_FILE_PATH}")
except Exception as e:
    print(f"\n寫入紀錄檔失敗： {e}")

print(f"\n所有實驗檔案已保存於: {RUN_DIR}")