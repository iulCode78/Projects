import React, { useState } from 'react';
import { Box, TextField, Button, Container, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import CircularProgress from '@mui/material/CircularProgress';


const SpotifyForm = () => {
    const [song, setSong] = useState('');
    const [Year, setYear] = useState('');
    const [songError,setSongError]=useState('');
    const [yearError, setYearError]=useState('');
    const navigate = useNavigate();
    
    const [load,setLoad] = useState(false)

    const [mappedArray, setMappedArray] = useState([])

    const validateForm = () =>{
        let isValid = true;

        // Validate song
        if (!song.trim()) {
            setSongError('Song name is required');
            isValid = false;
        } else {
            setSongError('');
        }

        // Validate Year
        if (!Year.trim()) {
            setYearError('Year is required');
            isValid = false;
        } else {
            setYearError('');
        }

        return isValid;
    }

    function processSongs(songsArray) {
        return songsArray.map(song => {
            // Process each song object as needed
            // For example, you might want to create a simpler object
            return {
                title: song.name,
                artist: song.artists,
                cover: song.albumCoverUrl,
                preview: song.trackPreviewUrl,
                trackURL: song.trackUrl
            };
        });
    }
    const handleButtonClickPost = async (song, year, onSuccess) => {
        try {
            // First, perform the POST request
            const dataToSend = { song, year };
            const postResponse = await fetch('http://localhost:5000/api/post', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(dataToSend),
            });
    
            if (!postResponse.ok) {
                throw new Error('POST request failed');
            }
    
            const postData = await postResponse.json();
            console.log('POST Success:', postData);
    
            // If POST is successful, perform the GET (fetch) request
            const fetchResponse = await fetch('http://localhost:5000/songs');
            if (!fetchResponse.ok) {
                throw new Error('Fetch request failed');
            }
    
            const fetchData = await fetchResponse.json();
            const data = fetchData
            // Update state
            
              // Create an array from the object's keys and values
            const mappedArray = processSongs(data)
            setMappedArray(data)
            
            console.log('Fetch Success:', fetchData);
            // Handle your fetch data here
            console.log('SONG ARRAY BELOW')
            console.log(mappedArray)
            if (onSuccess) {
                onSuccess(mappedArray); // Call the callback with the processed songs
            }
    
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const handleButtonClickFetch =() =>{
        fetch('http://localhost:5000/songs')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Success:', data);
            // Handle your data here
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });
    }

    // // Mock function to simulate getting recommended songs
    // const getRecommendedSongs = (song, Year) => {
    //     // This is a placeholder. Replace with actual logic or API call.
    //     return [
    //         { name: "Northern Attitude (With Hozier)", Year: "Noah Kahan" },
    //         { name: "Stick Season", Year: "Noah Kahan" },
    //         { name: "Come Over", Year: "Noah Kahan" },
    //         { name: "View Between Villages", Year: "Noah Kahan" },
    //         { name: "4runner", Year: "Brenn!" },
    //         { name: "Eat Your Young", Year: "Hozier" },
    //         { name: "Everywhere, Everything (with Grace Albrams)", Year: "Noah Kahan" },
    //     ];
    // };
    //When Submit Button Pressed Redirect to RecommendedSongs.js
    const handleSubmit = (e) => {
        e.preventDefault();
        const isValid = validateForm();
        if (!isValid) return;
        setLoad(true)
        // navigate('/recommendations', { state: { songs: recommendedSongs } });
         // Call 'handleButtonClickPost' and pass a callback function
        handleButtonClickPost(song, Year, (recommendedSongs) => {
        // Navigate after the songs data is fetched and processed
        navigate('/recommendations', { state: { songs: recommendedSongs } });
    });
        
    };
    const handleSetYear = (e) => {
        setYear(e)
        
    }

    return (
        <Container maxWidth="sm" sx={{ textAlign: 'center', mt: 4, fontFamily: "Arial, sans-serif" }}>
            <Box
                component="form"
                onSubmit={handleSubmit}
                noValidate
                sx={{
                    mt: 1,
                    backgroundColor: "#f0f8ff",
                    borderRadius: "15px",
                    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                    padding: "20px"
                }}
            >
                <img 
                    src="https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-icon-marilyn-scott-0.png" // Replace with your Spotify logo URL
                    alt="Spotify Logo"
                    style={{ maxWidth: '150px', margin: 'auto' }}
                />
                <br/>
                <Typography variant='h3'>Im in the mood for...</Typography>
                <TextField
                    margin="normal"
                    required
                    fullWidth
                    id="song"
                    label="Song Name"
                    variant='filled'
                    name="song"
                    autoComplete="song"
                    autoFocus
                    value={song}
                    error={!!songError}
                    helperText={songError}
                    onChange={(e) => setSong(e.target.value)}
                />
                {/*Text field for t*/}
                <TextField
                    margin="normal"
                    required
                    fullWidth
                    id="Year"
                    label="Year"
                    variant='filled'
                    name="Year"
                    autoComplete="Year"
                    value={Year}
                    error={!!yearError}
                    helperText={yearError}
                    onChange={(e) => handleSetYear(e.target.value)}
                />
                <Button
                    type="submit"
                    fullWidth
                    variant="contained"
                    sx={{ mt: 3, mb: 2 }}
                    onClick={() => handleButtonClickPost(song, Year)}
                >
                    Find Songs
                </Button>
                {load 
                ?    <CircularProgress color="secondary" />
                : <></>    
                }
            </Box>
                       {/* Text box below the form */}
                       <Box sx={{ marginTop: 2, backgroundColor: 'grey', padding: 2, borderRadius: '10px' }}>
                        <h4><p>Are you currently captivated by a particular song or Year? <br/>
                Longing to discover more tunes that resonate with the same vibe? <br/>
                Look no further,we have you covered! <br/>
                Simply enter your favorite song or Year, hit submit,  and let us curate a personalized playlist of similar tracks just for you. Immerse yourself in a world of music that mirrors your current obsession. Your next favorite song is just a click away!"


                </p></h4>
                {/* Add more paragraphs or other content as needed */}
            </Box>

        </Container>
    );
};



export default SpotifyForm;
