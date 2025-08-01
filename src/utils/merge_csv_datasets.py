import os
import pandas as pd
import re
from pathlib import Path
import logging
from typing import Dict, List, Tuple
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVMerger:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets-todos-links"
        self.output_dir = self.base_dir / "datasets-merged-2024-2025"
        self.source_dirs = {
            '2024': {
                'bbr': self.datasets_dir / "datasets-2024" / "originals" / "datasets vazao" / "bbr",
                'cubic': self.datasets_dir / "datasets-2024" / "originals" / "datasets vazao" / "cubic"
            },
            '2025': {
                'bbr': self.datasets_dir / "datasets-2025" / "originals" / "datasets vazao" / "bbr",
                'cubic': self.datasets_dir / "datasets-2025" / "originals" / "datasets vazao" / "cubic"
            }
        }
    def extract_base_name(self, filename: str) -> str:
        name_without_ext = filename.replace('.csv', '')
        date_pattern = r'\s+\d{2}-\d{2}-\d{4}$'
        base_name = re.sub(date_pattern, '', name_without_ext)
        return base_name.strip()
    def find_matching_files(self, algorithm: str) -> Dict[str, Dict[str, str]]:
        matching_files = {}
        dir_2024 = self.source_dirs['2024'][algorithm]
        dir_2025 = self.source_dirs['2025'][algorithm]
        if not dir_2024.exists():
            logger.warning(f"Diretório não encontrado: {dir_2024}")
            return matching_files
        if not dir_2025.exists():
            logger.warning(f"Diretório não encontrado: {dir_2025}")
            return matching_files
        files_2024 = {self.extract_base_name(f.name): f for f in dir_2024.glob('*.csv')}
        files_2025 = {self.extract_base_name(f.name): f for f in dir_2025.glob('*.csv')}
        common_base_names = set(files_2024.keys()) & set(files_2025.keys())
        for base_name in common_base_names:
            matching_files[base_name] = {
                '2024': str(files_2024[base_name]),
                '2025': str(files_2025[base_name])
            }
        return matching_files
    def merge_csv_files(self, file_2024: str, file_2025: str, output_file: str):
        try:
            df_2024 = pd.read_csv(file_2024)
            df_2025 = pd.read_csv(file_2025)
            if list(df_2024.columns) != list(df_2025.columns):
                logger.warning(f"Colunas diferentes entre arquivos: {file_2024} e {file_2025}")
                return False
            merged_df = pd.concat([df_2024, df_2025], ignore_index=True)
            if 'Data' in merged_df.columns:
                merged_df['Data'] = pd.to_datetime(merged_df['Data'], format='%d-%m-%Y', errors='coerce')
                merged_df = merged_df.dropna(subset=['Data'])
                merged_df = merged_df.sort_values('Data')
                merged_df['Data'] = merged_df['Data'].dt.strftime('%d-%m-%Y')
            merged_df.to_csv(output_file, index=False)
            logger.info(f"Arquivo mesclado criado: {output_file}")
            logger.info(f"Total de linhas: {len(merged_df)} (2024: {len(df_2024)}, 2025: {len(df_2025)})")
            return True
        except Exception as e:
            logger.error(f"Erro ao mesclar arquivos {file_2024} e {file_2025}: {e}")
            return False
    def merge_all_datasets(self):
        os.makedirs(self.output_dir, exist_ok=True)
        algorithms = ['bbr', 'cubic']
        total_merged = 0
        for algorithm in algorithms:
            logger.info(f"Processando algoritmo: {algorithm}")
            algorithm_output_dir = self.output_dir / algorithm
            os.makedirs(algorithm_output_dir, exist_ok=True)
            matching_files = self.find_matching_files(algorithm)
            if not matching_files:
                logger.warning(f"Nenhum arquivo compatível encontrado para {algorithm}")
                continue
            for base_name, files in matching_files.items():
                output_filename = f"{base_name} 2024-2025.csv"
                output_path = algorithm_output_dir / output_filename
                success = self.merge_csv_files(
                    files['2024'],
                    files['2025'],
                    str(output_path)
                )
                if success:
                    total_merged += 1
        logger.info(f"Processo concluído. Total de arquivos mesclados: {total_merged}")
def main():
    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    merger = CSVMerger(base_directory)
    merger.merge_all_datasets()
if __name__ == "__main__":
    main()
