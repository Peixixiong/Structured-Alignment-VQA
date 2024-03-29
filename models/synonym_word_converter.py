

#From composit word to single word
#Remove ' '
syn_dict_composit_multiwrds = {
    #Objects
        #Composit Words, some of them never show up in result, see object.split(',')[0]
    'alarm clock': 'clock',
    'ceiling fan': 'fan',
    'tail fin': 'tailfin',
    'birthday cake': 'cake',
    'stop sign': 'stop',
    'microwave oven': 'microwave',
    'skateboard ramp': 'ramp',
    'fridge': 'refrigerator',
    'knee pads': 'kneepad',
    'tennis court': 'field',
    'tv': 'television',
    'garage door': 'garagedoor',
    'sail boat': 'sailboat',
    'racquet': 'racket',
    'rock wall': 'wall',
    'head board': 'headboard',
    'tea kettle': 'kettle',
    'tennis racket': 'racket',
    'tennis racquet': 'racquet',
    'train station': 'station',
    'tennis player': 'player',
    'toilet brush':  'toiletbrush',
    'pepper shaker': 'peppershaker',
    'hair dryer': 'hairdryer',
    'toilet seat': 'toiletseat',
    'skate board': 'skateboard',
    'floor lamp': 'floorlamp',
    'french fries': 'fries',
    'christmas tree': 'christmas tree',
    'living room': 'livingroom',
    'teddybear': 'teddybear',
    'baseball field': 'field',
    'ski boot': 'skiboot',
    'shower curtain': 'curtain',
    'polar bear': 'polarbear',
    'hot dog': 'hotdog',
    'surf board': 'surfboard',
    'dirt bike': 'bike',
    'tail wing': 'tail',
    'area rug': 'rug',
    'bow tie': 'bowtie',
    'fire extinguisher': 'extinguisher',
    'tail feathers': 'feathers',
    'beach chair': 'chair',
    'fire hydrant': 'hydrant',
    'weather vane': 'weathercock',
    'soccer ball': 'soccer',
    'head band': 'headband',
    'bath tub': 'bathtub',
    'coffee table': 'coffeetable',
    'traffic light': 'trafficlight',
    'parking meter': 'parkingmeter',
    'wet suit': 'wetsuit',
    'teddy bears': 'teddybear',
    'suit case': 'suitcase',
    'tank top': 'tanktop',
    'shin guard': 'shinguard',
    'wii remote': 'wiiremote',
    'pizza slice': 'pizza',
    'home plate': 'homeplate',
    'ski boots': 'skiboots',
    'banana slice': 'bananaslice',
    'stuffed animals': 'stuffedanimals',
    'train platform': 'platform',
    'tissue box': 'tissuebox',
    'cutting board': 'cuttingboard',
    'license plate': 'licenseplate',
    'ski pole': 'skipole',
    'clock tower': 'clocktower',
    'toilet tank': 'toilettank',
    'palm trees': 'palmtrees',
    'skate park': 'skatepark',
    'computer monitor': 'monitor',
    'flip flop': 'slipper',
    'remote control': 'remotecontrol',
    'paper towels': 'papertowels',
    'train tracks': 'tracks',
    'soccer player': 'player',
    'doughnut': 'donut',
    'toilet bowl': 'toilet',
    'lounge chair': 'loungechair',
    'side walk': 'sidewalk',
    'tomato slice': 'tomatoslice',
    'window sill': 'windowsill',
    'toilet lid': 'toiletlid',
    "pitcher's mound": 'pitchermound',
    'palm tree': 'palmtree',
    'banana bunch': 'bananabunch',
    'tennis shoe': 'shoe',
    'giraffe head': 'giraffehead',
    'baseball player': 'player',
    'water bottle': 'bottle',
    'tennis ball': 'tennis',
    'cell phone': 'cellphone',
    'computer mouse': 'computermouse',
    'ski pants': 'skipants',
    'clock face': 'clock',
    'fire escape': 'fireescape',
    'police officer': 'police',
    'trash can': 'trashcan',
    'front window': 'window',
    'office chair': 'chair',
    'door knob': 'knob',
    'banana peel': 'bananapeel',
    'baseball game': 'baseballgame',
    'cabinet door': 'cabinetdoor',
    'night stand': 'nightstand',
    'traffic cone': 'trafficcone',
    'suit jacket': 'suit',
    'train engine': 'trainengine',
    'wrist band': 'wristband',
    'toilet paper': 'toiletpaper',
    'street sign': 'sign',
    'computer screen': 'screen',
    'wine glass': 'wineglass',
    'doughnuts': 'donuts',
    'train car': 'train',
    'tennis match': 'tennismatch',
    'railroad tracks': 'tracks',
    'stuffed bear': 'stuffedbear',
    'snow pants': 'pants',
    'neck tie': 'tie',
    'baseball bat': 'bat',
    'safety cone': 'safetycone',
    'paper towel': 'papertowel',
    'soccer field': 'field',
    'throw pillow': 'pillow',
    'oven door': 'ovendoor',
    'lamp shade': 'lampshade',
    'pine tree': 'pinetree',
    'lamp post': 'lamppost',
    'station wagon': 'car',
    'signal light': 'signallight',
    'american flag': 'flag',
    'baseball cap': 'cap',
    'front legs': 'legs',
    'life jacket': 'lifejacket',
    'water tank': 'watertank',
    'gas station': 'gasstation',
    'entertainment center': 'entertainment',
    'stuffed animal': 'stuffedanimal',
    'display case': 'displaycase',
    'front wheel': 'wheel',
    'coffee pot': 'coffeepot',
    'cowboy hat': 'hat',
    'table cloth': 'table cloth',
    'fire truck': 'firetruck',
    'game controller': 'gamecontroller',
    'sweat band': 'sweatband',
    'coin slot': 'coinslot',
    'pillow case': 'pillowcase',
    'coffee cup': 'cup',
    'counter top': 'countertop',
    'baseball uniform': 'baseballuniform',
    'book shelf': 'bookshelf',
    'facial hair': 'facialhair',
    'shin guards': 'shinguards',
    'tennis net': 'tennisnet',
    'trash bag': 'trashbag',
    'ski poles': 'skipoles',
    'gas tank': 'gastank',
    'soap dispenser': 'soapdispenser',
    'life vest': 'lifevest',
    'train front': 'trainfront',
    'exhaust pipe': 'pipe',
    'light fixture': 'light',
    'power lines': 'powerlines',
    'roman numerals': 'numbers',
    'picnic table': 'table',
    'wine bottle': 'winebottle',
    'tree trunk': 'trunk',
    'motor bike': 'motorcycle',
    'traffic sign': 'sign',
    'little girl': 'girl',
    'passenger car': 'passengercar',
    'brake light':'brakelight',
    'roman numeral':'number',
    'shower head':'showerhead',
    'handle bars': 'handlebars',
    'cardboard box': 'box',
    'mountain range': 'mountain',
    'eye glasses': 'glasses',
    'salt shaker': 'saltshaker',
    'knee pad': 'kneepad',
    'shower door':'showerdoor',
    'bathing suit':'bathingsuit',
    'manhole cover':'manholecover',
    'picture frame': 'pictureframe',
    'hour hand': 'hourhand',
    'dvd player': 'dvdplayer',
    'ski slope': 'slope',
    'french fry': 'fries',
    'landing gear': 'landinggear',
    'coffee maker': 'coffeemaker',
    'light switch': 'lightswitch',
    'tv stand':'tvstand',
    'steering wheel':'steeringwheel',
    'baseball glove':'baseballglove',
    'power pole':'telephonepole',
    'dirt road':'road',
    'telephone pole':'telephonepole',
    'tee shirt': 'tshirt',
    'face mask': 'facemask',
    'bathroom sink':'sink',
    'laptop computer':'laptop',
    'windshield wipers':'wipers',
    'tail light':'taillight',
    'snow board': 'snowboard',
    'stop light': 'stoplight',
    'ball cap': 'cap',
    'traffic signal': 'trafficsignal',
    'ski lift': 'skilift',
    'tennis shoes': 'shoes',
    'swim trunks': 'swimtrunks',
    'butter knife': 'knife',
    'train cars': 'trains',
    'pine trees': 'pinetres',
    'park bench': 'bench',
    'second floor': 'secondfloor',
    'hand towel': 'handtowel',
    'flip flops': 'slippers',
    'back pack': 'backpack',
    'ski tracks': 'tracks',
    'baseball players': 'players',
    'stone wall': 'wall',
    'dress shirt': 'shirt',
    'ski goggles': 'goggles',
    'power line': 'powerline',
    'train track': 'track',
    'air conditioner': 'air conditioner',
    'baseball mitt': 'mitt',
    'mouse pad': 'mousepad',
    'garbage can': 'trashcan',
    'taxi cab': 'taxi',
    'control panel': 'controlpanel',
    'clock hand': 'clockhand',
    'brick wall': 'wall',
    'grass field': 'field',
    'utility pole': 'telephonepole',
    'mountain top': 'montain',
    'hot dogs': 'hotdogs',
    'tail lights': 'taillights',
    'traffic lights': 'trafficlight',
    'candle holder': 'candleholder',
    'guard rail': 'guardrail',
    'tree branches': 'treebranches',
    'trash bin': 'trashcan',
    'side mirror': 'sidemirror',
    'street lamp': 'streetlamp',
    'paper plate': 'paperplate',
    'fence post': 'fence',
    'door frame': 'doorframe',
    'wire fence': 'fence',
    'table lamp': 'tablelamp',
    'pony tail': 'ponytail',
    'ocean water': 'ocean',
    'flower pot': 'flowerpot',
    'tree line': 'trees',
    'sign post': 'signpost',
    'passenger train': 'passengertrain',
    "catcher's mitt":'catchermitt',
    'electrical outlet': 'electricaloutlet',
    'bike rack': 'rack',
    'windshield wiper': 'windshieldwiper',
    'bus stop':'busstop',
    'police car': 'policecar',
    'name tag': 'nametag',
    'computer keyboard': 'computerkeyboard',
    'glass door': 'glassdoor',
    'wine glasses': 'wineglasses',
    'ski jacket':'jacket',
    'beer bottle':'bottle',
    'wrist watch': 'watch',
    'tile floor': 'tilefloor',
    'tree branch': 'treebranch',
    'towel rack' :'towelrack',

    #Attributes
    'long sleeved':'longsleeved',
    'light blue': 'blue',
    'light brown': 'brown',
    'partly cloudy': 'cloudy',
    'rainbow colored': 'rainbow',
    'half full': 'half',
    'having meeting':'meeting',

    #Relations
    'larger than': 'larger',
     'sitting by': 'sittingnear',
     'floating on': 'floating',
     'decorated with':'decorated',
     'riding in': 'riding',
     'sitting near':'sittingnear',
     'walking along':'walking',
     'on the back of':'back',
     'parked along': 'parked',
     'close to':'near',
     'parked by':'parked',
     'painted on':'painted',
     'on the front of':'front',
     'filled with': 'filled',
     'pushed by':'pushed',
     'scattered on':'scattered',
     'blowing out':'blowing',
     'printed on': 'printed',
     'worn on':'worn',
     'in front of':'front',
     'picking up':'picking',
     'pointing at':'pointing',
     'on the bottom of':'bottom',
     'staring at':'staring',
     'connected to':'connected',
     'chained to':'chained',
     'smaller than':'smaller',
     'to the left of':'left',
     'sprinkled on':'sprinkled',
     'surrounded by':'surrounded',
     'on top of':'top',
     'attached to':'attached',
     'to the right of':'right'





}
