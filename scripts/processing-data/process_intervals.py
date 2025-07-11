import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, time
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SixHourProcessor:
    def __init__(self, base_dir: str):

        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "datasets-merged-2024-2025"
        self.output_dir = self.base_dir / "intervalos-vazao"
        
        self.time_intervals = {
            'madrugada': (time(0, 0), time(5, 59, 59)),
            'manha': (time(6, 0), time(11, 59, 59)),
            'tarde': (time(12, 0), time(17, 59, 59)),
            'noite': (time(18, 0), time(23, 59, 59))
        }
        
    def get_time_interval(self, dt: datetime) -> str:
        current_time = dt.time()
        
        for interval_name, (start_time, end_time) in self.time_intervals.items():
            if start_time <= current_time <= end_time:
                return interval_name
        
        return 'unknown'
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("DataFrame vazio recebido")
            return pd.DataFrame(columns=['Data', 'Intervalo', 'Vazao'])
        
        df['Data'] = pd.to_datetime(df['Data'])
        
        df['Date'] = df['Data'].dt.date
        df['Hour'] = df['Data'].dt.hour
        
        interval_mapping = {
            'madrugada': '00:00:00 a 05:59:59',
            'manha': '06:00:00 a 11:59:59', 
            'tarde': '12:00:00 a 17:59:59',
            'noite': '18:00:00 a 23:59:59'
        }
        
        def categorize_interval(hour):
            if 0 <= hour < 6:
                return 'madrugada'
            elif 6 <= hour < 12:
                return 'manha'
            elif 12 <= hour < 18:
                return 'tarde'
            else:
                return 'noite'
        
        df['Time_Interval'] = df['Hour'].apply(categorize_interval)
        
        grouped = df.groupby(['Date', 'Time_Interval']).agg({
            'Vazao': 'mean'
        }).reset_index()
        
        all_dates = sorted(df['Date'].unique())
        
        result_data = []
        
        for date in all_dates:
            formatted_date = f"{date.day}-{date.month}-{date.year}"
            
            for interval_key, interval_text in interval_mapping.items():
                matching_data = grouped[
                    (grouped['Date'] == date) & 
                    (grouped['Time_Interval'] == interval_key)
                ]
                
                if len(matching_data) > 0:
                    vazao = round(matching_data['Vazao'].iloc[0], 2)
                else:
                    vazao = -1
                
                result_data.append({
                    'Data': formatted_date,
                    'Intervalo': interval_text,
                    'Vazao': vazao
                })
        
        result_df = pd.DataFrame(result_data)
        return result_df
    
    def process_file(self, input_file: Path, output_file: Path) -> bool:
        try:
            df = pd.read_csv(input_file)
            
            logger.info(f"Processando: {input_file.name}")
            logger.info(f"  Registros originais: {len(df)}")
            
            processed_df = self.process_dataframe(df)
            
            if processed_df.empty:
                logger.warning(f"  Nenhum dado processado para {input_file.name}")
                return False
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            processed_df.to_csv(output_file, index=False)
            
            logger.info(f"  Registros processados: {len(processed_df)}")
            logger.info(f"  Arquivo salvo: {output_file}")
            
            interval_stats = processed_df['Intervalo'].value_counts()
            vazao_stats = processed_df[processed_df['Vazao'] != -1]['Intervalo'].value_counts()
            logger.info(f"  Total de intervalos: {len(processed_df)}")
            logger.info(f"  Intervalos com dados: {len(vazao_stats)}")
            logger.info(f"  Intervalos sem dados (-1): {len(processed_df) - len(vazao_stats)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao processar {input_file}: {str(e)}")
            return False
    
    def create_output_filename(self, input_filename: str) -> str:
        return f"intervalos vazao {input_filename}"
    
    def process_algorithm(self, algorithm: str) -> Tuple[int, int]:

        logger.info(f"Processando algoritmo: {algorithm}")
        
        input_algorithm_dir = self.input_dir / algorithm
        output_algorithm_dir = self.output_dir / algorithm
        
        if not input_algorithm_dir.exists():
            logger.warning(f"Diretório não encontrado: {input_algorithm_dir}")
            return 0, 0
        
        output_algorithm_dir.mkdir(parents=True, exist_ok=True)
        
        successful_processes = 0
        failed_processes = 0
        
        for csv_file in input_algorithm_dir.glob("*.csv"):
            output_filename = self.create_output_filename(csv_file.name)
            output_path = output_algorithm_dir / output_filename
            
            success = self.process_file(csv_file, output_path)
            
            if success:
                successful_processes += 1
            else:
                failed_processes += 1
        
        logger.info(f"Algoritmo {algorithm} concluído:")
        logger.info(f"  - Sucessos: {successful_processes}")
        logger.info(f"  - Falhas: {failed_processes}")
        
        return successful_processes, failed_processes
    
    def generate_sample_analysis(self) -> None:

        logger.info("\n=== ANÁLISE DE EXEMPLO ===")
        
        sample_file = None
        for algorithm in ['bbr', 'cubic']:
            algorithm_dir = self.input_dir / algorithm
            if algorithm_dir.exists():
                csv_files = list(algorithm_dir.glob("*.csv"))
                if csv_files:
                    sample_file = csv_files[0]
                    break
        
        if not sample_file:
            logger.warning("Nenhum arquivo encontrado para análise de exemplo")
            return
        
        logger.info(f"Analisando arquivo: {sample_file.name}")
        
        df_original = pd.read_csv(sample_file)
        
        df_processed = self.process_dataframe(df_original.copy())
        
        df_original['Data'] = pd.to_datetime(df_original['Data'])
        
        sample_date = df_original['Data'].dt.date.value_counts().index[0]
        
        original_day = df_original[df_original['Data'].dt.date == sample_date]
        
        formatted_sample_date = f"{sample_date.day}-{sample_date.month}-{sample_date.year}"
        processed_day = df_processed[df_processed['Data'] == formatted_sample_date]
        
        logger.info(f"\nExemplo para o dia {sample_date}:")
        logger.info(f"Dados originais ({len(original_day)} registros):")
        for _, row in original_day.iterrows():
            logger.info(f"  {row['Data']} - Vazão: {row['Vazao']:,.0f}")
        
        logger.info(f"\nDados processados ({len(processed_day)} registros):")
        for _, row in processed_day.iterrows():
            vazao_str = str(row['Vazao']) if row['Vazao'] != -1 else "SEM DADOS"
            logger.info(f"  {row['Data']} {row['Intervalo']} - Vazão: {vazao_str}")
    
    def run(self) -> None:
        logger.info("Iniciando processamento de intervalos de 6 horas")
        logger.info(f"Diretório de entrada: {self.input_dir}")
        logger.info(f"Diretório de saída: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_success = 0
        total_failures = 0
        
        for algorithm in ['bbr', 'cubic']:
            success, failures = self.process_algorithm(algorithm)
            total_success += success
            total_failures += failures
        
        logger.info(f"Total de sucessos: {total_success}")
        logger.info(f"Total de falhas: {total_failures}")
        
        self.generate_sample_analysis()

def main():

    base_dir = "C:\\Users\\janaina\\OneDrive\\Documentos\\UECE-RNP-2024-REF"

    processor = SixHourProcessor(base_dir)
    
    processor.run()


if __name__ == "__main__":
    main()
