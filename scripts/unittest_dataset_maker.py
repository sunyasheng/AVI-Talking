import json
import random
import tqdm


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


train_sub_caps = [
    "A man feels fairly appalled and speaks with mildly lowered brow and marginally creased nose.                      ",
    "A very angry man speaks with extremely down brow.                                                                                   ",
    "An extremely scared man speaks with extremely raised inner brow, lip corner slightly pulled, upper lid quite lifted, and outer brow ",
    "An impassive man.                                                                                                                   ",
    "A fairly angry man speaks with brow fairly down and mildly raised inner brow.                                                       ",
    "A man feels slightly sad.                                                                                                           ",
    "A grouchy man speaks with quite down brow.                                                                                          ",
    "A man feels mildly sad and speaks with extremely down brow.                                                                         ",
    "A mildly joyous man speaks with lip corner lightly pulled.                                                                          ",
    "An impassive man.                                                                                                                   ",
    "A man is in shock and speaks with fully raised upper lid, outer brow significantly lifted, and fully raised inner brow.             ",
    "A mildly sad man speaks with marginally lifted inner brow and brow mildly down.                                                     ",
    "A fairly happy man speaks with extremely stretched lip corner.                                                                      ",
    "A man displays insecurity and speaks with mildly lifted upper lid, outer brow strongly lifted, and inner brow significantly lifted. ",
    "A gloomy man speaks with fairly raised cheek, pretty lowered brow, slightly raised inner brow, and mildly spread lip.               ",
    "A man feels displeased and speaks with brow quite down.                                                                             ",
    "A man displays rage and speaks with quite wrinkled nose and strongly lowered brow.                                                  ",
    "A fairly scornful man.                                                                                                              ",
    "A man feels disappointed and speaks with mildly down brow and marginally lifted inner brow.                                         ",
    "A man is in horror and speaks with mildly stretched lip corner, fully lifted inner brow, and pretty lifted outer brow.              ",
    "A man feels mildly sad and speaks with outer brow lightly raised and inner brow quite raised."
]

translated_caps = [
    "A man feels joy and speaks with a lowered brow, raised upper lip, separated lips, and dropped jaw.",
    "An angry man speaks with a lowered brow, raised cheek, wrinkled nose, and separated lips.",
    "A joyful man speaks with a raised cheek, pulled lip corner, and separated lips.",
    "A man feels grief and sadness and speaks with wrinkled nose, furrowed brows, raised upper lip, and inward pull of the cheeks, while slightly parted lips hint at annoyance and stress.",
    "A man feels anger or frustration and speaks with a downward pull of the mouth, raised cheeks, flared nostrils, and tension around the eyebrows.",
    "A happy man feels joy and speaks with raised cheeks, pulling of the lip corner, and separating of the lips.",
    "A man feels happiness, displaying a raised cheek, pulled lip corner, and separated lips.",
    "A man feels sadness and speaks with open eyes looking straight ahead, a closed or slightly open mouth, and a smooth forehead with minimal wrinkles.",
    "A man is emotionally detached and speaks with the chin being pulled downwards, strain on the neck muscles, squared-off jawline, and tightened lower lip.",
    "A fearful man speaks with the separation of the lips.",
    "A man feels sadness and speaks with raised inner eyebrow, separated lips, and dropped jaw.",
    "A happy man feels joy and speaks with open eyes, a slightly open or relaxed mouth position, and a relatively smooth forehead with minimal wrinkles.",
    "A man feels happiness, with separated lips indicating joy.",
    "A man feels joy and happiness, with visible lip movement, the upper lip pulled upwards towards the nose, stretching and wrinkling of the surrounding skin, and slightly widened nostrils.",
    "A man feels joy, with raised eyebrows, wrinkles around the eyes, a squinted nose, lowered upper lip, and tightened cheeks, speaking with a smile.",
    "A neutral man is inferred, with lips pulled apart and slightly open, corners of the mouth pulled downwards, chin pushed upwards, skin around the eyes pulled.",
    "A happy man speaks with partially closed eyes, lifted cheeks, and mouth drawn back with tightened or pursed lips.",
    "A man expresses happiness, with raised eyebrows, wrinkles across the forehead, open wider than usual eyes with crow's feet forming, lifted and puffed-out cheeks, and stretched lips.",
    "A man shows a state of calmness and contentment, with a relaxed expression, slightly open or closed eyes, a smooth forehead, and minimal wrinkles.",
    "A calm and neutral man speaks with a relatively smooth forehead and relaxed eyes and mouth, indicating a state of calmness and contentment.",
]

def format_mead_text():
    audio_path = '/data/yashengsun/local_storage/Mead_emoca/Mead_W/W019_front_angry_level2_007/W019_front_angry_level2_007.wav'
    json_path = 'audio_instruction_meadtext.json'
    global train_sub_caps
    audio_paths = [audio_path] * len(train_sub_caps)
    train_sub_caps = [cap.split('.')[0] for cap in train_sub_caps]
    # import pdb; pdb.set_trace()
    pair_dict = {'text_descs': train_sub_caps, 'audio_paths': audio_paths}
    write_json_file(json_path, pair_dict)


def format_celebv_text():
    audio_path = '/data/yashengsun/local_storage/Mead_emoca/Mead_W/W019_front_angry_level2_007/W019_front_angry_level2_007.wav'
    json_path = 'audio_instruction_celebvtext.json'
    global translated_caps
    audio_paths = [audio_path] * len(translated_caps)
    translated_caps = [cap.split('.')[0] for cap in translated_caps]
    # import pdb; pdb.set_trace()
    pair_dict = {'text_descs': translated_caps, 'audio_paths': audio_paths}
    write_json_file(json_path, pair_dict)


def format_exp_blip():
    gen_expblip_path = '/data/yashengsun/Proj/TalkingFace/exp_blip/captions/All_train_facial_expression_captions.json'
    audio_path = '/data/yashengsun/local_storage/Mead_emoca/Mead_W/W019_front_angry_level2_007/W019_front_angry_level2_007.wav'

    gen_expblip_all = read_json_file(gen_expblip_path)
    gen_expblip_select = random.sample(gen_expblip_all, 50)
    
    audio_paths = [audio_path] * len(gen_expblip_select)
    text_descs = [dict_i['caption'] for dict_i in gen_expblip_select]
    # import pdb; pdb.set_trace()

    json_path = 'audio_instruction.json'
    pair_dict = {'text_descs': text_descs, 'audio_paths': audio_paths}
    write_json_file(json_path, pair_dict)


def display_audio(json_path):
    audio_json = read_json_file(json_path)
    for text_desc in audio_json['text_descs'][:20]:
        print(text_desc)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # format_mead_text()
    format_celebv_text()
    # main()
    # json_path = 'audio_instruction.json'
    # display_audio(json_path)
