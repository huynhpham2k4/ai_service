import pandas as pd
import numpy as np

# Mapping từ 61 phoneme sang 39 phoneme
phon61_map39 = {
    'iy':'iy',  'ih':'ih',   'eh':'eh',  'ae':'ae',    'ix':'ih',  'ax':'ah',   'ah':'ah',  'uw':'uw',
    'ux':'uw',  'uh':'uh',   'ao':'aa',  'aa':'aa',    'ey':'ey',  'ay':'ay',   'oy':'oy',  'aw':'aw',
    'ow':'ow',  'l':'l',     'el':'l',  'r':'r',      'y':'y',    'w':'w',     'er':'er',  'axr':'er',
    'm':'m',    'em':'m',     'n':'n',    'nx':'n',     'en':'n',  'ng':'ng',   'eng':'ng', 'ch':'ch',
    'jh':'jh',  'dh':'dh',   'b':'b',    'd':'d',      'dx':'dx',  'g':'g',     'p':'p',    't':'t',
    'k':'k',    'z':'z',     'zh':'sh',  'v':'v',      'f':'f',    'th':'th',   's':'s',    'sh':'sh',
    'hh':'hh',  'hv':'hh',   'pcl':'h#', 'tcl':'h#', 'kcl':'h#', 'qcl':'h#','bcl':'h#','dcl':'h#',
    'gcl':'h#','h#':'h#',  '#h':'h#',  'pau':'h#', 'epi': 'h#','nx':'n',   'ax-h':'ah','q':'h#' 
}

def map_phonemes(phoneme_sequence):
    # Tách chuỗi phoneme thành các phoneme riêng lẻ
    phonemes = phoneme_sequence.split()
    # Map từng phoneme và giữ lại số stress (0,1,2)
    mapped_phonemes = []
    for p in phonemes:
        # Tách phoneme và stress
        if len(p) > 2 and p[-1] in '012':
            base_phoneme = p[:-1]
            stress = p[-1]
            mapped_phoneme = phon61_map39.get(base_phoneme, base_phoneme) 
        else:
            mapped_phoneme = phon61_map39.get(p, p)
        mapped_phonemes.append(mapped_phoneme)
    return ' '.join(mapped_phonemes)

def clean_audio_data(input_file, output_file):
    # Đọc file CSV
    df = pd.read_csv(input_file)
    
    # Map phonemes từ 61 sang 39
    df['phoneme'] = df['phoneme'].apply(map_phonemes)
    
    # Lưu file đã được xử lý
    df.to_csv(output_file, index=False)
    print(f"Đã xử lý và lưu file thành công vào {output_file}")
    print(f"Số hàng sau khi xử lý: {len(df)}")

if __name__ == "__main__":
    input_file = "../data_craw/audio_data.csv"
    output_file = "../data_craw/audio_data_cleaned.csv"
    clean_audio_data(input_file, output_file) 