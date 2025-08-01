import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import glob
import logging

warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
    
    def configure_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                tf.config.optimizer.set_jit(True) 
                tf.config.experimental.enable_tensor_float_32()           
                return True
            except Exception as e:
                print(f"Erro na configuração da GPU: {e}")
                return False
        return False
    
    GPU_AVAILABLE = configure_gpu()
    
except ImportError:
    KERAS_AVAILABLE = False
    GPU_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    
    try:
        xgb.train({'tree_method': 'gpu_hist', 'gpu_id': 0}, 
                 xgb.DMatrix([[1]], label=[1]), num_boost_round=1, verbose_eval=False)
        XGBOOST_GPU_AVAILABLE = True
        print("XGBoost com suporte a GPU disponível!")
    except:
        XGBOOST_GPU_AVAILABLE = False
        print("XGBoost disponível, mas sem suporte a GPU")
        
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_data(self, sample_files=None):
        logger.info("Carregando dados...")
        all_data = []
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                logger.warning(f"Diretório não encontrado: {data_dir}")
                continue
                
            csv_files = self._get_csv_files(data_dir, sample_files)
            
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    df['fonte'] = os.path.basename(file_path)
                    df['diretorio'] = os.path.basename(data_dir)
                    all_data.append(df)
                    logger.info(f"Carregado: {file_path}")
                except Exception as e:
                    logger.error(f"Erro ao carregar {file_path}: {e}")
        
        if not all_data:
            raise ValueError("Nenhum dado foi carregado. Verifique os diretórios e arquivos.")
        
        self.data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total de registros carregados: {len(self.data)}")
    
    def _get_csv_files(self, data_dir, sample_files):
        if sample_files:
            return [os.path.join(data_dir, f) for f in sample_files 
                   if os.path.exists(os.path.join(data_dir, f))]
        return glob.glob(os.path.join(data_dir, "*.csv"))
        
    def preprocess_data(self):
        logger.info("Pré-processando dados...")
        
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        self._convert_datetime()
        self._create_time_features()
        self._map_intervals()
        self._remove_outliers()
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        logger.info("Pré-processamento concluído.")
    
    def _convert_datetime(self):
        self.data['Data'] = pd.to_datetime(self.data['Data'], format='%d-%m-%Y')
    
    def _create_time_features(self):
        self.data['ano'] = self.data['Data'].dt.year
        self.data['mes'] = self.data['Data'].dt.month
        self.data['dia'] = self.data['Data'].dt.day
        self.data['dia_semana'] = self.data['Data'].dt.dayofweek
    
    def _map_intervals(self):
        interval_mapping = {
            '00:00:00 a 05:59:59': 0,
            '06:00:00 a 11:59:59': 6,
            '12:00:00 a 17:59:59': 12,
            '18:00:00 a 23:59:59': 18
        }
        self.data['hora_inicio'] = self.data['Intervalo'].map(interval_mapping)
        self.data['timestamp'] = self.data['Data'] + pd.to_timedelta(self.data['hora_inicio'], unit='h')
    
    def _remove_outliers(self):
        Q1 = self.data['Vazao'].quantile(0.25)
        Q3 = self.data['Vazao'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_outliers = len(self.data)
        self.data = self.data[(self.data['Vazao'] >= lower_bound) & (self.data['Vazao'] <= upper_bound)]
        after_outliers = len(self.data)
        logger.info(f"Outliers removidos: {before_outliers - after_outliers}")
        
    def create_features(self, lookback_days=7):
        logger.info(f"Criando features com lookback de {lookback_days} dias...")
        
        feature_data = []
        
        for fonte in self.data['fonte'].unique():
            df_fonte = self.data[self.data['fonte'] == fonte].copy()
            df_fonte = df_fonte.sort_values('timestamp').reset_index(drop=True)
            
            df_fonte = self._create_lag_features(df_fonte, lookback_days)
            df_fonte = self._create_rolling_features(df_fonte)
            df_fonte = self._create_cyclical_features(df_fonte)
            
            feature_data.append(df_fonte)
        
        self.data = pd.concat(feature_data, ignore_index=True)
        
        initial_rows = len(self.data)
        self.data = self.data.dropna().reset_index(drop=True)
        final_rows = len(self.data)
        logger.info(f"Linhas removidas devido a NaN: {initial_rows - final_rows}")
        logger.info("Features criadas com sucesso.")
    
    def _create_lag_features(self, df, lookback_days):
        for lag in range(1, lookback_days * 4 + 1):
            df[f'vazao_lag_{lag}'] = df['Vazao'].shift(lag)
        return df
    
    def _create_rolling_features(self, df):
        for window in [4, 8, 12, 24]:
            df[f'vazao_ma_{window}'] = df['Vazao'].shift(1).rolling(window=window).mean()
            df[f'vazao_std_{window}'] = df['Vazao'].shift(1).rolling(window=window).std()
        return df
    
    def _create_cyclical_features(self, df):
        df['hora_sin'] = np.sin(2 * np.pi * df['hora_inicio'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora_inicio'] / 24)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        return df
        
    def prepare_ml_data(self):
        logger.info("Preparando dados para ML...")
        
        feature_cols = [col for col in self.data.columns if 
                       col.startswith(('vazao_lag_', 'vazao_ma_', 'vazao_std_')) or
                       col.endswith(('_sin', '_cos')) or
                       col in ['hora_inicio', 'dia_semana', 'mes', 'ano']]
        
        X = self.data[feature_cols].copy()
        y = self.data['Vazao'].copy()
        
        logger.info(f"Features selecionadas: {len(feature_cols)}")
        logger.info(f"Amostras totais: {len(X)}")
        
        return X, y
        
    def calculate_metrics(self, y_true, y_pred):
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'SMAPE': 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        }
        
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        metrics['Theil_U'] = numerator / denominator if denominator != 0 else np.inf
        
        if len(y_true) > 1:
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['DA'] = np.mean(actual_direction == pred_direction) * 100
        else:
            metrics['DA'] = np.nan
            
        return metrics
        
    def detect_data_leakage(self, model_name, metrics):
        suspicious = False
        warnings = []
        
        if metrics['R2'] > 0.99:
            warnings.append(f"R² suspeito: {metrics['R2']:.6f}")
            suspicious = True
            
        if hasattr(self, 'data') and metrics['RMSE'] < self.data['Vazao'].std() * 0.01:
            warnings.append(f"RMSE muito baixo: {metrics['RMSE']:.2f} vs std: {self.data['Vazao'].std():.2f}")
            suspicious = True
            
        if metrics['MAPE'] < 0.1:
            warnings.append(f"MAPE suspeito: {metrics['MAPE']:.4f}%")
            suspicious = True
            
        if suspicious:
            logger.warning(f"POSSÍVEL VAZAMENTO DE DADOS detectado em {model_name}:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
            
        return suspicious
        
    def train_traditional_models(self, X, y, test_size=0.2):
        logger.info("Treinando modelos tradicionais...")
        
        data_with_features = pd.concat([X, y], axis=1)
        data_with_features = data_with_features.sort_index()
        
        split_idx = int(len(data_with_features) * (1 - test_size))
        
        X_train = data_with_features.iloc[:split_idx][X.columns]
        X_test = data_with_features.iloc[split_idx:][X.columns]
        y_train = data_with_features.iloc[:split_idx]['Vazao']
        y_test = data_with_features.iloc[split_idx:]['Vazao']
        
        logger.info(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        if XGBOOST_AVAILABLE:
            if XGBOOST_GPU_AVAILABLE:
                models['XGBoost_GPU'] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='rmse'
                )
                logger.info("XGBoost com GPU adicionado aos modelos!")
            else:
                models['XGBoost_CPU'] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    tree_method='hist',
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='rmse'
                )
                logger.info("XGBoost (CPU) adicionado aos modelos!")
        
        for name, model in models.items():
            logger.info(f"Treinando {name}...")
            
            if name in ['LinearRegression', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif 'XGBoost' in name:
                model.fit(X_train, y_train, 
                         eval_set=[(X_test, y_test)], 
                         verbose=False)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            self.detect_data_leakage(name, metrics)
            
            self.models[name] = model
            self.results[name] = {
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }
            
            logger.info(f"{name} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.4f}")
    
    def train_lstm_model(self, X, y, test_size=0.2, sequence_length=24, use_gpu=True):
        if not KERAS_AVAILABLE:
            logger.warning("TensorFlow/Keras não disponível. LSTM será omitido.")
            return
            
        logger.info("Treinando modelo LSTM...")
        
        device_name = '/GPU:0' if use_gpu and GPU_AVAILABLE else '/CPU:0'
        logger.info(f"Treinando LSTM na {'GPU' if 'GPU' in device_name else 'CPU'}...")
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, sequence_length)
        
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        model = self._build_lstm_model(sequence_length, X.shape[1], device_name)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]
        
        if GPU_AVAILABLE and use_gpu:
            batch_size = 128  
            epochs = 150
            logger.info(f"Configuração otimizada para GPU: batch_size={batch_size}, epochs={epochs}")
        else:
            batch_size = 32  
            epochs = 100
            logger.info(f"Configuração para CPU: batch_size={batch_size}, epochs={epochs}")
        
        with tf.device(device_name):
            logger.info(f"Iniciando treinamento LSTM no dispositivo: {device_name}")
            
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1 if GPU_AVAILABLE else 0,
                use_multiprocessing=True if not GPU_AVAILABLE else False,
                workers=4 if not GPU_AVAILABLE else 1
            )
            
            logger.info("Gerando predições...")
            y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0)
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        metrics = self.calculate_metrics(y_test_orig, y_pred)
        
        self.models['LSTM'] = model
        self.scalers['lstm_X'] = scaler_X
        self.scalers['lstm_y'] = scaler_y
        self.results['LSTM'] = {
            'metrics': metrics,
            'y_test': y_test_orig,
            'y_pred': y_pred,
            'history': history
        }
        
        logger.info(f"LSTM - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.4f}")
    
    def _create_sequences(self, X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def _build_lstm_model(self, sequence_length, n_features, device_name):
        with tf.device(device_name):
            if 'GPU' in device_name and GPU_AVAILABLE:
                try:
                    from keras.layers import CuDNNLSTM
                    lstm_layer = CuDNNLSTM
                    logger.info("Usando CuDNNLSTM para máxima performance na GPU!")
                except ImportError:
                    lstm_layer = LSTM
                    logger.info("Usando LSTM padrão")
            else:
                lstm_layer = LSTM
            
            model = Sequential([
                lstm_layer(64, return_sequences=True, input_shape=(sequence_length, n_features)),
                Dropout(0.3),
                lstm_layer(32, return_sequences=True),
                Dropout(0.3),
                lstm_layer(16, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            if 'GPU' in device_name and GPU_AVAILABLE:
                try:
                    from keras.mixed_precision import Policy
                    policy = Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Mixed precision (float16) habilitado para GPU!")
                except:
                    pass
                
                optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            else:
                optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def check_gpu_status(self):
        
        if not KERAS_AVAILABLE:
            print("TensorFlow não está instalado")
            return False
            
        print(f"TensorFlow versão: {tf.__version__}")
        
        physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        
        print(f"GPUs físicas detectadas: {len(physical_gpus)}")
        print(f"GPUs lógicas configuradas: {len(logical_gpus)}")
        
        if physical_gpus:
            for i, gpu in enumerate(physical_gpus):
                print(f"\nGPU {i}: {gpu.name}")
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if 'device_name' in gpu_details:
                        print(f"   Nome: {gpu_details['device_name']}")
                    if 'compute_capability' in gpu_details:
                        print(f"   Compute Capability: {gpu_details['compute_capability']}")
                except:
                    print("   Detalhes não disponíveis")
            
            print(f"\nTeste rápido de GPU...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("GPU está funcionando corretamente!")
                
                # Verificar memória da GPU
                try:
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    current_mb = gpu_info['current'] / (1024**2)
                    peak_mb = gpu_info['peak'] / (1024**2)
                    print(f"Memória GPU - Atual: {current_mb:.1f} MB | Pico: {peak_mb:.1f} MB")
                except:
                    pass
                
                return True
            except Exception as e:
                print(f"Erro ao testar GPU: {e}")
                return False
        else:
            print("Nenhuma GPU detectada")
            
        print("\nDICAS PARA USAR GPU:")
        print("   - Instale CUDA Toolkit 11.8 ou 12.x")
        print("   - Instale cuDNN compatível")
        print("   - Use: pip install tensorflow[and-cuda]")
        print("   - Para XGBoost: pip install xgboost[gpu]")
        print("=" * 60)
        
        return False
    
    def cross_validate_models(self, X, y, n_splits=5):
        logger.info(f"Executando validação cruzada com {n_splits} folds...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        if XGBOOST_AVAILABLE:
            if XGBOOST_GPU_AVAILABLE:
                models['XGBoost_GPU'] = xgb.XGBRegressor(
                    n_estimators=100, random_state=42, tree_method='gpu_hist', 
                    gpu_id=0, eval_metric='rmse'
                )
            else:
                models['XGBoost_CPU'] = xgb.XGBRegressor(
                    n_estimators=100, random_state=42, tree_method='hist', 
                    eval_metric='rmse'
                )
        
        for name, model in models.items():
            fold_metrics = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                if name == 'LinearRegression':
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_val_fold = scaler.transform(X_val_fold)
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                metrics = self.calculate_metrics(y_val_fold, y_pred)
                fold_metrics.append(metrics)
            
            cv_results[name] = {}
            for metric in fold_metrics[0].keys():
                values = [fold[metric] for fold in fold_metrics if not np.isnan(fold[metric])]
                cv_results[name][f'{metric}_mean'] = np.mean(values)
                cv_results[name][f'{metric}_std'] = np.std(values)
            
            logger.info(f"{name} CV - RMSE: {cv_results[name]['RMSE_mean']:.2f} ± {cv_results[name]['RMSE_std']:.2f}")
        
        return cv_results
    
    def plot_results(self, save_path=None):
        logger.info("Gerando gráficos...")
        
        if not self.results:
            logger.warning("Nenhum resultado para plotar.")
            return
        
        self._plot_metrics_comparison(save_path)
        self._plot_predictions(save_path)
    
    def _plot_metrics_comparison(self, save_path):
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        models = list(self.results.keys())
        metrics_names = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for i, metric in enumerate(metrics_names):
            if i < len(axes):
                values = [self.results[model]['metrics'][metric] for model in models]
                axes[i].bar(models, values)
                axes[i].set_title(f'{metric} por Modelo')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'model_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_predictions(self, save_path):
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['metrics']['RMSE'])
        
        plt.figure(figsize=(15, 8))
        y_test = self.results[best_model]['y_test']
        y_pred = self.results[best_model]['y_pred']
        
        plt.subplot(2, 1, 1)
        plt.plot(y_test.values[:100], label='Real', alpha=0.7)
        plt.plot(y_pred[:100], label='Predito', alpha=0.7)
        plt.title(f'Predições vs Real - {best_model} (primeiras 100 amostras)')
        plt.legend()
        plt.ylabel('Vazão')
        
        plt.subplot(2, 1, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')
        plt.title(f'Scatter Plot - {best_model}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'predictions_{best_model}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        logger.info("=== RESUMO DOS RESULTADOS ===")
        
        if not self.results:
            logger.info("Nenhum modelo foi treinado.")
            return
        
        self._print_gpu_usage_summary()
        self._print_metrics_table()
        self._print_best_model()
        self._print_data_statistics()
    
    def _print_gpu_usage_summary(self):
        print("\nRESUMO DE USO DA GPU:")
        print("-" * 50)
        
        gpu_models = [name for name in self.results.keys() if 'GPU' in name or name == 'LSTM']
        cpu_models = [name for name in self.results.keys() if name not in gpu_models]
        
        if gpu_models:
            print(f"Modelos executados na GPU: {', '.join(gpu_models)}")
        if cpu_models:
            print(f"Modelos executados na CPU: {', '.join(cpu_models)}")
            
        if GPU_AVAILABLE:
            print("Status: GPU ativa e otimizada")
        else:
            print("Status: Executando apenas em CPU")
            
        print(f"TensorFlow: {'GPU habilitado' if KERAS_AVAILABLE and GPU_AVAILABLE else 'Apenas CPU'}")
        print(f"XGBoost: {'GPU disponível' if XGBOOST_GPU_AVAILABLE else 'CPU apenas'}")
        print("-" * 50)
    
    def _print_metrics_table(self):
        print("\nMétricas por Modelo:")
        print("-" * 80)
        print(f"{'Modelo':<15} {'RMSE':<12} {'MAE':<12} {'R2':<8} {'MAPE':<8} {'SMAPE':<8}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            print(f"{model_name:<15} {metrics['RMSE']:<12.2f} {metrics['MAE']:<12.2f} "
                  f"{metrics['R2']:<8.4f} {metrics['MAPE']:<8.2f} {metrics['SMAPE']:<8.2f}")
    
    def _print_best_model(self):
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['metrics']['RMSE'])
        print(f"\nMelhor modelo (menor RMSE): {best_model}")
        print(f"RMSE: {self.results[best_model]['metrics']['RMSE']:.2f}")
        print(f"R²: {self.results[best_model]['metrics']['R2']:.4f}")
    
    def _print_data_statistics(self):
        print(f"\nEstatísticas dos dados:")
        print(f"Total de amostras: {len(self.data)}")
        print(f"Período: {self.data['Data'].min().strftime('%d/%m/%Y')} a {self.data['Data'].max().strftime('%d/%m/%Y')}")
        print(f"Número de conexões: {self.data['fonte'].nunique()}")
        print(f"Vazão média: {self.data['Vazao'].mean():.2f}")
        print(f"Vazão mediana: {self.data['Vazao'].median():.2f}")
        print(f"Desvio padrão: {self.data['Vazao'].std():.2f}")

def main():
    data_directories = [
        r"C:\Users\janaina\OneDrive\Documentos\UECE-RNP-2024-REF\intervalos-vazao\svd\bbr-imputed\svd",
        r"C:\Users\janaina\OneDrive\Documentos\UECE-RNP-2024-REF\intervalos-vazao\svd\cubic-imputed\svd"
    ]
    
    predictor = TimeSeriesPredictor(data_directories)
    gpu_status = predictor.check_gpu_status()
    
    try:
        predictor.load_data()
        predictor.preprocess_data()
        predictor.create_features(lookback_days=5)
        
        X, y = predictor.prepare_ml_data()
        
        predictor.train_traditional_models(X, y)
        
        if KERAS_AVAILABLE:
            logger.info("Iniciando treinamento LSTM...")
            predictor.train_lstm_model(X, y, sequence_length=12, use_gpu=gpu_status)
        
        cv_results = predictor.cross_validate_models(X, y)
        
        predictor.print_summary()
        predictor.plot_results(save_path=os.path.dirname(__file__))
        
        logger.info("Pipeline concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main()
