import logo from './logo.svg';
import './App.css';
import React from "react";
import Calendar from 'react-calendar'
import 'react-calendar/dist/Calendar.css';

import Timer from './Timer.js';
import './todo.js';
import Todo from './todo.js';

import BreakTimer from './BreakTimer.js';



function App()
{
  return (
    <div className="App">
      <header className="App-header">
      <img src={logo} className="App-logo" alt="logo" />
      <div className="content-container">
        <div className="row">            
          <div className="left-panel box">
            <Quotes></Quotes>
            <div id="container-buttons">
              <button class="button button1">Upload Notes</button>
              <button class="button button2">View All Notes</button>
            </div>
                     
          </div>            
          <div className="middle-panel box">
            <div className="calendar">
            <Timer></Timer>
            <BreakTimer></BreakTimer>
            <br></br>
            <Calendar></Calendar>
            

            </div>
          </div>            
          <div className="right-panel box">
              <div className = "Spotify">
                <a href="https://open.spotify.com/playlist/37i9dQZF1DX8NTLI2TtZa6" target="_blank" rel="noreferrer">
                  <img
                    src = "https://blog.trello.com/hubfs/download%20%2889%29.png"
                    width="150"
                    height="100"
                    alt="headphones.png"
                  />
                </a>
                  <h5>Intense Studying Spotify Playlist</h5>
                  <div> <br></br></div>
                  <Todo></Todo>
              </div>
          </div>       
        </div>
      </div>
      </header>
    </div>
  );
}

export default App;

class Quotes extends React.Component {
  constructor(props) {
    super(props);
    this.switchImage = this.switchImage.bind(this);
    this.state = {
      currentImage: 0,
      images: [
        "https://cdn.shopify.com/s/files/1/0070/7032/files/Fearless_Motivational_Quote_Desktop_Wallpaper_1.png?format=webp&v=1600450412",
        "https://cdn.shopify.com/s/files/1/0070/7032/files/Nelsson_Mandela.png?format=webp&v=1600448850",
        "https://www.success.com/wp-content/uploads/legacy/sites/default/files/new2.jpg",
        "https://cdn.lifehack.org/wp-content/uploads/2022/06/motivational_quotes_11.jpg",
        "https://www.success.com/wp-content/uploads/legacy/sites/default/files/15_15.jpg",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/proud-quote-1522094934.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/right-quote-1522095339.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/impossible-quote-1522095589.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/stand-up-quote-1522095787.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/comes-easy-quote-1522095903.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/victory-quote-1522095975.png",
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/doubt-quote-1522096094.png",
        "https://img.freepik.com/premium-vector/inspirational-motivation-quotes-harder-you-work-better-you-get_67445-139.jpg?w=2000",
      ]
    };
  }

  switchImage() {
    if (this.state.currentImage < this.state.images.length - 1) {
      this.setState({
        currentImage: this.state.currentImage + 1
      });
    } else {
      this.setState({
        currentImage: 0
      });
    }
    return this.currentImage;
  }

  componentDidMount() {
    setInterval(this.switchImage, 10000);
  }

  render() {
    return (
      <div className="quotes">
        <img
          src={this.state.images[this.state.currentImage]} width={500} height={250} 
          alt="cleaning images"
        />
      </div>
    );
  }
}

/*future useful links
ON-CLICK - calls function when item is clicked
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjNoMDBiOj6AhWKQEEAHWMHBCwQFnoECAkQAw&url=https%3A%2F%2Fupmostly.com%2Ftutorials%2Freact-onclick-event-handling-with-examples&usg=AOvVaw0c4hEM-rGE9SNDCtdwF-6Z

Spilt page and have separate divs down side
https://towardsdev.com/putting-components-side-by-side-in-react-for-beginners-39a4ed7ab03b

src="https://www.clipartmax.com/png/middle/351-3519663_headphones-png-icon-svg-stock-headset-png-icon.png"

https://www.freecodecamp.org/news/how-to-build-a-react-project-with-create-react-app-in-10-steps/

REFERENCES
https://stackoverflow.com/questions/57107633/changing-an-image-on-in-time-interval-using-react
https://stackoverflow.com/questions/56470223/how-to-resize-img-in-react
https://www.npmjs.com/package/react-calendar
*/