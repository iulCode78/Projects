// RecommendedSongs.js
import React from 'react';
import { useLocation } from 'react-router-dom';
//import { Button } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import { Button } from '@mui/material';


const RecommendedSongs = () => {
    const location = useLocation();
    console.log('checking location', location.state)
    const songs = location.state?.songs || [];
    console.log(songs)

    const formatDuration = (duration) => {
        const minutes = Math.floor(duration / 60000);
        const seconds = ((duration % 60000) / 1000).toFixed(0);
        return `${minutes}:${seconds.padStart(2, '0')}`;
    };

    return (
        <div>
            <h1 style={{textAlign:'center'}}>Recommended Songs</h1>
            {songs.map((song, index) => (
                <div key={index} style={songCardStyle}>
                    <img src={song.cover} alt={song.name} style={albumArtStyle} />
                    <div style={songDetailsStyle}>
                        <p><strong>{song.title}</strong> </p>
                        <p>Artists: {song.artist}</p>
                        {song.preview ? (
                            <p>Track Preview: <audio controls src={song.preview} /></p>
                        ) : (
                            <p>No track preview available</p>
                        )}
                    </div>
                    <div style={buttonGroupStyle}>
                    <Button 
                        variant='contained' 
                        onClick={() => window.open(song.trackURL, '_blank')}
                    >
                        <PlayArrowIcon style={{ fontSize: '50px', marginRight: '10px' }} />
                    </Button>
                    </div>
                </div>
            ))}
        </div>
    );
};

const songCardStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '20px',
    padding: '10px',
    backgroundColor: '#f5f5f5', // Adjust the background color as needed
    borderRadius: '8px'
};

const albumArtStyle = {
    width: '100px',
    height: '100px',
    marginRight: '20px'
};

const songDetailsStyle = {
    flexGrow: 1
};

const buttonGroupStyle = {
    display: 'flex',
    gap: '10px'
};

export default RecommendedSongs;

