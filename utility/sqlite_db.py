import sqlite3


class FaceEmotionDB:
    def __init__(self, db_name="sql/faces_emotions.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create face table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS face (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            image_path TEXT
        )
        ''')

        # Create face_emotions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            image_path TEXT,
            face_id INTEGER,
            angry_conf INTEGER,
            fear_conf INTEGER,
            neutral_conf INTEGER,
            sad_conf INTEGER,
            disgust_conf INTEGER,
            happy_conf INTEGER,
            surprise_conf INTEGER,
            dominant_emotion TEXT,
            FOREIGN KEY(face_id) REFERENCES face(id)
        )
        ''')
        self.conn.commit()



    def update_face_name(self, face_id, new_name):
        self.cursor.execute('''
        UPDATE face SET name = ? WHERE id = ?
        ''', (new_name, face_id))
        self.conn.commit()

    def insert_face_unknown(self, face_data):
        # Insert face data with 'unknown' as name placeholder
        self.cursor.execute('''
        INSERT INTO face (name, x1, y1, x2, y2, image_path)
        VALUES ('unknown', ?, ?, ?, ?, ?)
        ''', (face_data['x1'], face_data['y1'], face_data['x2'], face_data['y2'], face_data['image_path']))
        self.conn.commit()
        row_id = self.cursor.lastrowid
        self.update_face_name(row_id, f'unknown_{row_id}')
        return row_id

    def insert_face(self, face_data):
        self.cursor.execute('''
        INSERT INTO face (name, x1, y1, x2, y2, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            face_data['name'], face_data['x1'], face_data['y1'], face_data['x2'], face_data['y2'],
            face_data['image_path']))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_emotion(self, emotion_data):
        self.cursor.execute('''
        INSERT INTO face_emotions (x1, y1, x2, y2, image_path, face_id, angry_conf, fear_conf, neutral_conf, sad_conf, disgust_conf, happy_conf, surprise_conf, dominant_emotion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            emotion_data['x1'], emotion_data['y1'], emotion_data['x2'], emotion_data['y2'], emotion_data['image_path'],
            emotion_data['face_id'], emotion_data['angry_conf'], emotion_data['fear_conf'],
            emotion_data['neutral_conf'],
            emotion_data['sad_conf'], emotion_data['disgust_conf'], emotion_data['happy_conf'],
            emotion_data['surprise_conf'], emotion_data['dominant_emotion']))
        self.conn.commit()

    def fetch_all_faces(self):
        self.cursor.execute('SELECT * FROM face')
        return self.cursor.fetchall()

    def fetch_all_face_data(self):
        self.cursor.execute('SELECT id, name, image_path FROM face')
        return self.cursor.fetchall()

    def fetch_emotions_for_face(self, face_id):
        self.cursor.execute('SELECT * FROM face_emotions WHERE face_id = ?', (face_id,))
        return self.cursor.fetchall()

    def fetch_all_emotions_with_faces(self):
        self.cursor.execute(
            'SELECT face.*, face_emotions.* FROM face JOIN face_emotions ON face.id = face_emotions.face_id')
        return self.cursor.fetchall()

    def fetch_all_emotion_data(self):
        self.cursor.execute('''
        SELECT 
            face.name, 
            face_emotions.dominant_emotion, 
            face_emotions.angry_conf, 
            face_emotions.fear_conf, 
            face_emotions.neutral_conf, 
            face_emotions.sad_conf, 
            face_emotions.disgust_conf, 
            face_emotions.happy_conf, 
            face_emotions.surprise_conf,
            face_emotions.image_path
        FROM face_emotions 
        JOIN face ON face.id = face_emotions.face_id
        ''')
        return self.cursor.fetchall()



    def close(self):
        self.conn.close()
