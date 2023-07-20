
def main():
    styleA = '''Style A: 
A man feels fairly appalled and speaks with mildly lowered brow and marginally creased nose.                      
A very angry man speaks with extremely down brow.                                                                                   
An extremely scared man speaks with extremely raised inner brow, lip corner slightly pulled, upper lid quite lifted, and outer brow 
An impassive man.                                                                                                                   
A fairly angry man speaks with brow fairly down and mildly raised inner brow.                                                       
A man feels slightly sad.                                                                                                           
A grouchy man speaks with quite down brow.                                                                                          
A man feels mildly sad and speaks with extremely down brow.                                                                         
A mildly joyous man speaks with lip corner lightly pulled.                                                                          
An impassive man.                                                                                                                   
A man is in shock and speaks with fully raised upper lid, outer brow significantly lifted, and fully raised inner brow.             
A mildly sad man speaks with marginally lifted inner brow and brow mildly down.                                                     
A fairly happy man speaks with extremely stretched lip corner.                                                                      
A man displays insecurity and speaks with mildly lifted upper lid, outer brow strongly lifted, and inner brow significantly lifted. 
A gloomy man speaks with fairly raised cheek, pretty lowered brow, slightly raised inner brow, and mildly spread lip.               
A man feels displeased and speaks with brow quite down.                                                                             
A man displays rage and speaks with quite wrinkled nose and strongly lowered brow.                                                  
A fairly scornful man.                                                                                                              
A man feels disappointed and speaks with mildly down brow and marginally lifted inner brow.                                         
A man is in horror and speaks with mildly stretched lip corner, fully lifted inner brow, and pretty lifted outer brow.              
A man feels mildly sad and speaks with outer brow lightly raised and inner brow quite raised.'''

    styleB = '''Style B:
The facial actions of lowering the brow, raising the upper lip, separating the lips, and dropping the jaw all point to a smiling expression, thus inferring the emotion of happiness.
The anger is inferred from the lowered brow, raised cheek, wrinkled nose, and separated lips of this person's face.
This happy face is shown by the raising of the cheek, the pulling of the lip corner, and the separation of the lips.
The wrinkling of the nose, furrowed brows, and raised upper lip along with the upward and inward pull of the cheeks and narrowed eyes suggest grief and sadness, while the slightly parted lips hint at annoyance and stress in this expression.
These facial actions indicate that the person is experiencing anger or frustration, as evidenced by the downward pull of the mouth, raised cheeks, flared nostrils, and tension around the eyebrows.
This face shows happiness, as it is inferred from the raising of the cheeks, pulling of the lip corner and separating of the lips.
The raised cheek, pulled lip corner, and separated lips infer a sense of happiness in this person's expression.
The eyes being open and looking straight ahead and the mouth being closed or slightly open in a relaxed position along with the smooth forehead with minimal wrinkles or gaps between the brain and the eyeballs and slightly closed or slightly opened lower eyelids with partially or completely concealed eyelashes, indicate a sad emotion on this face.
The facial actions, including the chin being pulled downwards and the strain on the neck muscles, as well as the squared off jawline and tightened lower lip, suggest a sense of emotional detachment, in line with the expression of neutral that is observed on this face.
This fearful expression is conveyed through the separation of the lips.
The person's facial actions, consisting of raising their inner eyebrow, separating their lips, and dropping their jaw, suggest a feeling of sadness.
The eyes being open and looking straight ahead, with a slightly open or relaxed mouth position and a relatively smooth forehead with minimal wrinkles or spacing between the forehead and nose, along with slightly lifted or rounded cheeks and minimal wrinkles or space between the cheeks and nose, all suggest that the person is feeling happy.
The separated lips in this face indicate happiness.
The visible lip movement involving the upper lip being pulled upwards towards the nose and exposing the upper teeth, as well as the stretching and wrinkling of the surrounding skin and slightly widened nostrils, all indicate joy and happiness. The slight protruding of the chin due to the muscular tension further supports this emotion.
The raised eyebrows and wrinkles around the eyes indicate joy, while the squinted nose and lowered upper lip reveal a sense of contentment, and the tightened cheeks suggest a smile of happiness.
The face with lips pulled apart and slightly open, corners of the mouth pulled downwards, chin pushed upwards, skin around the eyes pulled, cheeks slightly bulged out, some visible dimpling on the face, and eyes appearing more open or wider is inferred to show neutral.
From the facial actions of eyes partially closed, cheeks lifted, and mouth drawn back with tightened or pursed lips, it can be inferred that the emotion being expressed in this face is happiness.
The combination of raised eyebrows, wrinkles across the forehead, open wider than usual eyes with crow's feet forming at the corners, lifted and puffed out cheeks, laterally stretched and pulled back lips with no teeth or gums exposed, as well as deep vertical creases present on either side of the mouth extending from the nose to the chin, and the lower eyelid showing tension, all suggest that the emotion being expressed is happiness.
The relaxed position of the slightly open or closed eyes and mouth, along with the smooth forehead and minimal wrinkles, suggest a state of calmness and contentment, which is indicative of the emotion of neutrality.
The relatively smooth forehead and relaxed position of the eyes and mouth suggests a calm and neutral expression. There are minimal wrinkles or gaps between the eyes, which further highlights the lack of emotion conveyed through this facial expression.
'''

    conversion_prompt = 'Translate every sentence in style B to Style A.\n{}\n\n{}'.format(styleA, styleB)
    print(conversion_prompt)


if __name__ == '__main__':
    main()