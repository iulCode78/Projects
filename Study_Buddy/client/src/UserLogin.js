import React from 'react';
import './UserLogin.css';

function UserLogin() {
  return (
    <form>
      <h1>264 is ass :)</h1>
      <div className="container">
        <div className="form-group">
          <h3>Username</h3>
          <input type="text" name="name" id="name"/>
        </div>
        <div className="form-group">
          <h3>Password</h3>
          <input type="password" name="password" id="password"/>
        </div>
        <input type="submit" value="Login"/>
      </div>
    </form>
  )
}

export default UserLogin;
