
import aiohttp
import asyncio
import uvicorn
import os
import requests
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
Port = int(os.environ.get('PORT', 50000))
export_file_url = 'https://drive.google.com/uc?export=download&id=1Rgg1zbJjpfAWp7sjayWaSV4d8WQO2DPn'
export_file_name = 'export.pkl'

classes = ['Adhirasam', 'Aloo gobi', 'Aloo matar', 'Aloo methi',
       'Aloo shimla mirch', 'Aloo tikki', 'Alu Pitika', 'Amti', 'Anarsa',
       'Ariselu', 'Attu', 'Avial', 'Bajri no rotlo', 'Balu shahi',
       'Bandar laddu', 'Basundi', 'Bebinca', 'Beef Fry', 'Bengena Pitika',
       'Bhakri', 'Bhatura', 'Bhindi masala', 'Bilahi Maas', 'Biryani',
       'Bisi bele bath', 'Black rice', 'Bombil fry', 'Boondi',
       'Bora Sawul', 'Brown Rice', 'Butter chicken', 'Chak Hao Kheer',
       'Chakali', 'Cham cham', 'Chana masala', 'Chapati', 'Cheera Doi',
       'Chevdo', 'Chhena jalebi', 'Chhena kheeri', 'Chhena poda',
       'Chicken Tikka', 'Chicken Tikka masala', 'Chicken Varuval',
       'Chicken razala', 'Chikki', 'Chingri Bhape', 'Chingri malai curry',
       'Chole bhature', 'Chorafali', 'Churma Ladoo', 'Coconut vadi',
       'Copra paak', 'Currivepillai sadam ', 'Daal Dhokli',
       'Daal baati churma', 'Daal puri', 'Dahi vada', 'Dal makhani ',
       'Dal tadka', 'Dalithoy', 'Dharwad pedha', 'Dhokla', 'Dhondas',
       'Doodhpak', 'Dosa', 'Double ka meetha', 'Dudhi halwa', 'Dum aloo',
       'Fara', 'Farsi Puri', 'Gajar ka halwa', 'Galho', 'Gatta curry',
       'Gavvalu', 'Gheela Pitha', 'Ghevar', 'Ghooghra', 'Goja',
       'Gud papdi', 'Gulab jamun', 'Halvasan', 'Hando Guri', 'Handwo',
       'Haq Maas', 'Idiappam', 'Idli', 'Imarti', 'Jalebi', 'Jeera Aloo',
       'Kaara kozhambu', 'Kabiraji', 'Kachori', 'Kadai paneer',
       'Kadhi pakoda', 'Kajjikaya', 'Kaju katli', 'Kakinada khaja',
       'Kalakand', 'Kanji', 'Kansar', 'Karela bharta', 'Keerai kootu',
       'Keerai masiyal', 'Keerai poriyal', 'Keerai sadam', 'Keri no ras',
       'Khakhra', 'Khaman', 'Khandvi', 'Khar', 'Kheer', 'Kheer sagar',
       'Khichdi', 'Khichu', 'Khorisa', 'Kofta', 'Koldil Chicken',
       'Koldil Duck', 'Kolim Jawla', 'Kombdi vade', 'Konir Dom', 'Kootu',
       'Kos kootu', 'Koshambri', 'Koshimbir', 'Kothamali sadam',
       'Kulfi falooda', 'Kumol Sawul', 'Kutchi dabeli', 'Kuzhakkattai',
       'Kuzhambu', 'Kuzhi paniyaram', 'Laapsi', 'Laddu', 'Lassi',
       'Lauki ke kofte', 'Lauki ki subji', 'Ledikeni', 'Lilva Kachori',
       'Litti chokha', 'Luchi', 'Lyangcha', 'Maach Jhol', 'Mag Dhokli',
       'Mahim halwa', 'Makki di roti sarson da saag', 'Malapua',
       'Masala Dosa', 'Masor Koni', 'Masor tenga', 'Mawa Bati',
       'Methi na Gota', 'Mihidana', 'Mishti Chholar Dal', 'Misi roti',
       'Misti doi', 'Modak', 'Mohanthal', 'Mushroom do pyaza',
       'Mushroom matar', 'Muthiya', 'Mysore pak', 'Naan', 'Namakpara',
       'Nankhatai', 'Navrattan korma', 'Obbattu holige', 'Pachadi',
       'Pakhala', 'Palak paneer', 'Palathalikalu', 'Paneer butter masala',
       'Paneer tikka masala', 'Pani Pitha', 'Pani puri', 'Paniyaram',
       'Panjeeri', 'Pantua', 'Papad', 'Papadum', 'Paratha', 'Paravannam',
       'Paruppu sadam', 'Patra', 'Pattor', 'Pav Bhaji', 'Payasam',
       'Payokh', 'Pesarattu', 'Petha', 'Phirni', 'Pinaca', 'Pindi chana',
       'Pithe', 'Poha', 'Pongal', 'Poornalu', 'Pootharekulu', 'Poriyal',
       'Pork Bharta', 'Prawn malai curry', 'Puli sadam', 'Puri Bhaji',
       'Puttu', 'Qubani ka meetha', 'Rabri', 'Rajma chaval', 'Ras malai',
       'Rasabali', 'Rasam', 'Rasgulla', 'Red Rice', 'Rongi', 'Saath',
       'Sabudana Khichadi', 'Sambar', 'Samosa', 'Sandesh', 'Sandige',
       'Sattu ki roti', 'Sev khamani', 'Sev tameta', 'Sevai',
       'Shahi paneer', 'Shahi tukra', 'Shankarpali', 'Sheer korma',
       'Sheera', 'Shrikhand', 'Shufta', 'Shukto', 'Singori',
       'Sohan halwa', 'Sohan papdi', 'Sukhdi', 'Surnoli', 'Sutar feni',
       'Tandoori Chicken', 'Tandoori Fish Tikka', 'Thalipeeth',
       'Thayir sadam', 'Theeyal', 'Thepla', 'Til Pitha',
       'Turiya Patra Vatana sabji', 'Undhiyu', 'Unni Appam', 'Upma',
       'Uttapam', 'Vada', 'Veg Kolhapuri', 'Vegetable jalfrezi',
       'Vindaloo', 'Zunka']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=Port, log_level="info")
