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
        
        files_2024 = {}
        for file_path in dir_2024.glob("*.csv"):
            base_name = self.extract_base_name(file_path.name)
            files_2024[base_name] = str(file_path)
        
        for file_path in dir_2025.glob("*.csv"):
            base_name = self.extract_base_name(file_path.name)
            if base_name in files_2024:
                matching_files[base_name] = {
                    '2024': files_2024[base_name],
                    '2025': str(file_path)
                }
        
        logger.info(f"Encontrados {len(matching_files)} pares de arquivos para {algorithm}")
        return matching_files
    
    def merge_csv_files(self, file_2024: str, file_2025: str, output_path: str) -> bool:
        try:
            df_2024 = pd.read_csv(file_2024)
            df_2025 = pd.read_csv(file_2025)
            
            if list(df_2024.columns) != list(df_2025.columns):
                logger.warning(f"Colunas diferentes encontradas em {file_2024} e {file_2025}")
                common_columns = list(set(df_2024.columns) & set(df_2025.columns))
                if common_columns:
                    df_2024 = df_2024[common_columns]
                    df_2025 = df_2025[common_columns]
                else:
                    logger.error(f"Nenhuma coluna comum encontrada entre {file_2024} e {file_2025}")
                    return False
            
            merged_df = pd.concat([df_2024, df_2025], ignore_index=True)
            
            if 'Timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('Timestamp').reset_index(drop=True)
            elif 'timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            merged_df.to_csv(output_path, index=False)
            
            logger.info(f"Arquivo unido salvo: {output_path}")
            logger.info(f"  - Linhas 2024: {len(df_2024)}")
            logger.info(f"  - Linhas 2025: {len(df_2025)}")
            logger.info(f"  - Total: {len(merged_df)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao unir {file_2024} e {file_2025}: {str(e)}")
            return False
    
    def create_output_filename(self, base_name: str, algorithm: str) -> str:
        return f"{base_name} 2024-2025.csv"
    
    def process_algorithm(self, algorithm: str) -> None:
        logger.info(f"Processando algoritmo: {algorithm}")
        matching_files = self.find_matching_files(algorithm)
        
        if not matching_files:
            logger.warning(f"Nenhum arquivo correspondente encontrado para {algorithm}")
            return
        
        output_algorithm_dir = self.output_dir / algorithm
        output_algorithm_dir.mkdir(parents=True, exist_ok=True)
        
        successful_merges = 0
        failed_merges = 0
        
        for base_name, files in matching_files.items():
            output_filename = self.create_output_filename(base_name, algorithm)
            output_path = output_algorithm_dir / output_filename
            
            success = self.merge_csv_files(
                files['2024'],
                files['2025'],
                str(output_path)
            )
            
            if success:
                successful_merges += 1
            else:
                failed_merges += 1
        
        logger.info(f"Algoritmo {algorithm} concluído:")
        logger.info(f"  - Sucessos: {successful_merges}")
        logger.info(f"  - Falhas: {failed_merges}")
    
    def run(self) -> None:
        logger.info("Iniciando processo de união de arquivos CSV")
        logger.info(f"Diretório base: {self.base_dir}")
        logger.info(f"Diretório de saída: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for algorithm in ['bbr', 'cubic']:
            self.process_algorithm(algorithm)
        
        logger.info("Processo de união concluído!")
        

def main():
 
    base_dir = "C:\\Users\\janaina\\OneDrive\\Documentos\\UECE-RNP-2024-REF"
    
    merger = CSVMerger(base_dir)
    merger.run()


if __name__ == "__main__":
    main()
