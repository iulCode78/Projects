import React from 'react';
import './Timer.css';

function padTime(time) {
  return time.toString().padStart(2, "0");
}

function Timer() {
  const [timeleft, setTimeLeft] = React.useState(2700);
  const [running, setRunning] = React.useState(false);

  const minutes = padTime(Math.floor(timeleft / 60));
  const seconds = padTime((timeleft - (minutes * 60)));
  const interval = React.useRef(null);

  function startTimer() {

    setRunning(true);

    interval.current = setInterval(() => {
      setTimeLeft(timeleft => {
        if (timeleft >= 1) {
          return timeleft - 1;
        } else {
          return 0;
        }
      });
    }, 1000);
  }

  function stopTimer() {
    clearInterval(interval.current)
    setRunning(false);
  }

  function resetTimer() {
    clearInterval(interval.current);
    setTimeLeft(2700);
    setRunning(false);
  }

  return (
    <div className="count">
      <br></br>
      <h4><b>Study Timer</b></h4>
      <div className="stopwatch">
        <span>{minutes}</span>
        <span>:</span>
        <span>{seconds}</span>
      </div>
      <div className="press">
        
        {!running && <button id = "timerbutton" onClick={startTimer}>Start</button>}
        <br />
        {running && <button id = "timerbutton" onClick={stopTimer}>Stop</button>}
        {!running && <button id = "timerbutton" onClick={resetTimer}>Reset</button>}
      </div>
    </div>
  );
}

export default Timer;

