// Footer.js
import React from 'react';

const Footer = () => {
  return (
    <footer style={footerStyle}>
      <p>© 2024 Melody Match <br/> <i>Ciaran Doherty | Cillian Deegan | Jack Malone | Iulian Boariu</i></p>
      {/* Add more elements as needed */}
    </footer>
  );
};

const footerStyle = {
  backgroundColor: '#333',
  color: 'white',
  textAlign: 'center',
  padding: '10px 0',
  position: 'sticky', //jack: keep this sticky or it blocks part of webpage
  left: 0,
  bottom: 0,
  width: '100%'
};

export default Footer;
