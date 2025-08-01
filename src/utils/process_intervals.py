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
        def assign_interval(hour):
            if 0 <= hour <= 5:
                return interval_mapping['madrugada']
            elif 6 <= hour <= 11:
                return interval_mapping['manha']
            elif 12 <= hour <= 17:
                return interval_mapping['tarde']
            elif 18 <= hour <= 23:
                return interval_mapping['noite']
            else:
                return 'unknown'
        df['Interval'] = df['Hour'].apply(assign_interval)
        grouped = df.groupby(['Date', 'Interval'])['Vazao'].mean().reset_index()
        grouped['Data'] = grouped['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))
        grouped = grouped.rename(columns={'Interval': 'Intervalo'})
        result = grouped[['Data', 'Intervalo', 'Vazao']].copy()
        result = result.sort_values(['Data', 'Intervalo'])
        return result
    def process_single_file(self, input_file: Path, output_file: Path) -> bool:
        try:
            logger.info(f"Processando arquivo: {input_file.name}")
            df = pd.read_csv(input_file)
            if df.empty:
                logger.warning(f"Arquivo vazio: {input_file}")
                return False
            processed_df = self.process_dataframe(df)
            if processed_df.empty:
                logger.warning(f"Nenhum dado processado para: {input_file}")
                return False
            output_file.parent.mkdir(parents=True, exist_ok=True)
            processed_df.to_csv(output_file, index=False)
            logger.info(f"Arquivo processado salvo: {output_file}")
            logger.info(f"Linhas: {len(df)} -> {len(processed_df)}")
            return True
        except Exception as e:
            logger.error(f"Erro ao processar {input_file}: {e}")
            return False
    def process_all_files(self):
        if not self.input_dir.exists():
            logger.error(f"Diretório de entrada não encontrado: {self.input_dir}")
            return
        total_processed = 0
        total_files = 0
        for algorithm_dir in self.input_dir.iterdir():
            if not algorithm_dir.is_dir():
                continue
            algorithm_name = algorithm_dir.name
            logger.info(f"Processando algoritmo: {algorithm_name}")
            output_algorithm_dir = self.output_dir / algorithm_name
            for csv_file in algorithm_dir.glob('*.csv'):
                total_files += 1
                output_filename = f"intervalos vazao {csv_file.name}"
                output_file = output_algorithm_dir / output_filename
                if self.process_single_file(csv_file, output_file):
                    total_processed += 1
        logger.info(f"Processamento concluído: {total_processed}/{total_files} arquivos processados")
def main():
    import os
    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processor = SixHourProcessor(base_directory)
    processor.process_all_files()
if __name__ == "__main__":
    main()
