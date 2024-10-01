// Header.js
import React from 'react';
import { Link } from 'react-router-dom';
import HomeIcon from '@mui/icons-material/Home'

const Header = () => {
  return (
    <header style={headerStyle}>
      <Link to='/' style={iconStyle}>
        <HomeIcon style={{color:'white', fontSize:'45px'}}/>
        </Link>
      <h1 style={{textAlign:'center', flexGrow:1}}>Melody Match</h1>
    </header>
  );
};

const headerStyle = {
  backgroundColor: '#333',
  color: 'white',
  padding: '10px 0',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center'
};

const iconStyle = {
  color: 'white',
  padding: '10px',
  marginLeft: '30px'
}

export default Header;
