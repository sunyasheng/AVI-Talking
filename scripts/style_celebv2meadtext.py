
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

    styleB = '''Style B sentences:
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

    # conversion_prompt = 'Translate every sentence in style B to Style A.\n{}\n\n{}'.format(styleA, styleB)
    # print(conversion_prompt)

    conversion_prompt = '''
Style B sentences:
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

Summarized Style B sentences with one of following structures:  

A man feels _ and speaks with _ and _.                      
A _ man speaks with _.                                                                                   
A _ man speaks with _, _, _, and _.
A _ man.
A man feels _.
A man feels _ and speaks with _.
A man displays _ and speaks with _.
A man is _ and speaks with _.

Could use the following emotions and actions:
emotions:

angry:
  level1:
    feel: [annoyed, irritated, displeased, mildly angry, slightly angry]
    attr: [grouchy, grumpy, mildly angry, slightly angry]
  level2:
    feel: [angry, resentful, frustrated, bitter, fairly angry, quite angry]
    show: [hatred, anger, resentment, hostility]
    attr: [angry, hostile, fairly angry, quite angry]
  level3:
    feel: [exasperated, outraged, rage, outraged, furious, extremely angry, very angry]
    show: [exasperation, rage, outrage, fury, wrath, loathing]
    attr: [outraged, wrathful, hysterical, extremely angry, very angry]


contempt:
  level1:
    attr: [mildly contemptuous, slightly contemptuous, mildly scornful, slightly scornful, mildly disdainful, slightly disdainful]
  level2:
    attr: [fairly contemptuous, quite contemptuous, fairly scornful, quite scornful, fairly disdainful, quite disdainful]
  level3:
    attr: [extremely contemptuous, very contemptuous, extremely scornful, very scornful, extremely disdainful, very disdainful]


disgusted:
  level1:
    feel: [mildly disgusted, slightly disgusted, mildly appalled, slightly appalled, mildly sickened, slightly sickened]
    attr: [mildly disgusted, slightly disgusted, mildly appalled, slightly appalled, mildly sickened, slightly sickened]
  level2:
    feel: [disgusted, fairly disgusted, quite disgusted, fairly appalled, quite appalled, fairly sickened, quite sickened]
    show: [scorn, dislike]
    attr: [disgusted, fairly disgusted, quite disgusted, fairly appalled, quite appalled, fairly sickened, quite sickened]
  level3:
    feel: [revulsion, extremely disgusted, very disgusted, extremely appalled, very appalled, extremely sickened, very sickened]
    show: [hatred, loathing, revulsion]
    is_in: [revulsion]
    attr: [vengeful, extremely disgusted, very disgusted, extremely appalled, very appalled, extremely sickened, very sickened]



fear:
  level1:
    feel: [insecure, anxious, nervous, tense, mildly scared, slightly scared, mildly frightened, slightly frightened]
    show: [insecurity, anxiety]
    attr: [insecure, anxious, nervous, tense, mildly scared, slightly scared, mildly frightened, slightly frightened]
  level2:
    feel: [fear, frightened, uneasy, worried, apprehensive, fairly scared, quite scared, fairly frightened, quite frightened]
    show: [worry, apprehension]
    attr: [frightened, uneasy, worried, apprehensive, fairly scared, quite scared, fairly frightened, quite frightened]
  level3:
    feel: [horrified, terror, panic, extremely scared, very scared, extremely frightened, very frightened]
    is_in: [horror, panic]
    attr: [horrified, extremely scared, very scared, extremely frightened, very frightened]


happy:
  level1: 
    feel: [satisfied, content, mildly happy, slightly happy, mildly joyous, slightly joyous, mildly delighted, slightly delighted]
    show: [satisfaction, contentment]
    attr: [mildly happy, slightly happy, mildly joyous, slightly joyous, mildly delighted, slightly delighted]
  level2:
    feel: [cheerful, delighted, glad, enthusiastic, pleased, jolly, fairly happy, quite happy, fairly joyous, quite joyous, fairly delighted, quite delighted]
    show: [eagerness, enthusiasm]
    attr: [cheerful, joyous, delighted, eager, enthusiastic, jovial, fairly happy, quite happy, fairly joyous, quite joyous, fairly delighted, quite delighted]
  level3:
    feel: [ecstatic, jubilant, elated, euphoric, excited, thrilled, exhilarated, extremely happy, very happy, extremely joyous, very joyous, extremely delighted, very delighted]
    show: [adoration, affection, love, fondness, excitement]
    attr: [jubilant, ecstatic, euphoric, excited, exhilarated, extremely happy, very happy, extremely joyous, very joyous, extremely delighted, very delighted]


sad:
  level1:
    feel: [mildly sad, slightly sad]
    attr: [mildly sad, slightly sad]
  level2:
    feel: [gloomy, unhappy, disappointed, fairly sad, quite sad]
    attr: [gloomy, unhappy, disappointed, glum, fairly sad, quite sad]
  level3:
    feel: [tormented, anguished, sorrowful, miserable, desperate, hopeless, grieved, extremely sad, very sad]
    is_in: [agony, anguish, grief, sorrow, despair, distress]
    show: [sorrow, woe, melancholy, despair, grief]
    attr: [tormented, anguished, sorrowful, miserable, desperate, hopeless, extremely sad, very sad]


surprised:
  level1:
    feel: [mildly surprised, slightly surprised]
    attr: [mildly surprised, slightly surprised]
  level2:
    feel: [fairly surprised, quite surprised]
    is_in: [alarm]
    attr: [fairly surprised, quite surprised]
  level3:
    feel: [shocked, amazed, astonished, extremely surprised, very surprised]
    is_in: [shock]
    attr: [shocked, astonished, extremely surprised, very surprised]


neutral:
  level1:
    attr: [impassive, calm, expressionless]


actions:

Adj+Noun,raised inner brow,raised outer brow,lowered brow,raised upper lid,raised cheek,tightened lid,wrinkled nose,raised upper lip,deepened nasolabial,pulled lip corner,puffed cheek,dimpled cheek,depressed lip corner,depressed lower lip,raised chin,puckered lip,stretched lip,tightened neck,funneled lip,tightened lip,pressed lip,parted lips,droped jaw,stretched mouth,sucked lip,thrust jaw,sideways jaw,clenched jaw,bit lip,blown cheek,puffed cheek,sucked cheek,bulged tongue,wiped lip,dilated nostril,compressed nostril,sniffing nose,drooped lid,slit eyes,closed eyes,squint eye,,,,,,,,,,,,,,,
Noun,inner brow,outer brow,brow,upper lid,cheek,lid,nose,upper lip,nasolabial,lip corner,cheek,cheek,lip corner,lower lip,chin,lip,lip,neck,lip,lip,lip,lips,jaw,mouth,lip,jaw,jaw,jaw,lip,cheek,cheek,cheek,tongue,lip,nostril,nostril,nose,lid,eyes,eyes,eye,eyes,eye,head,head,head,head,head,head,head,head,eyes,eyes,eyes,eyes,
Adj1,raised,raised,lowered,raised,raised,tightened,wrinkled,raised,deepened,pulled,puffed,dimpling,depressed,depressed,raised,puckered,stretched,tightened,funneled,tightened,pressed,parted,dropped,stretched,sucked,thrust,sideways,clenched,bit,blown,puffed,sucked,bulged,wiped,dilated,compressed,sniffing,drooped,slit,closed,squint,blinking,winking,turned left,turned right,up,down,tilted left,tilted right,forward,back,turned left,turned right,up,down,getting walleyed
Adj2,lifted,lifted,down,lifted,lifted,narrowed,creased,lifted,,stretched,blown,,pushed down,pushed down,lifted,pursy,spread,constricted,,narrowed,pushed,separated,fallen,spread,pulled,pushed,,gritted,nipped,puffed,blown,drawn,swollen,,inflated,narrowed,snuffled,sagged,leering,shut,narrowed,,,moving left,moving right,raised,lowered,leaned left,leaned right,forth,backward,moving left,moving right,raised,lowered,

'''



if __name__ == '__main__':
    main()