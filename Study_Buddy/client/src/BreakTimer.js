import React from 'react';
import './BreakTimer.css';

function padTime(time) {
  return time.toString().padStart(2, "0");
}

function BreakTimer() {
  const [timeleft, setTimeLeft] = React.useState(900);
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
    setTimeLeft(900);
    setRunning(false);
  }

  return (
    <div className="count">
      <h4><b>Break Timer</b></h4>
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

export default BreakTimer;