import React from 'react';
import Carosel from './Carosel';
import Card from './Card';
import Card1 from './Card1';
import Card2 from './Card2';
import makeup from '../images/makeup (1).jpeg'
import background from '../images/makeup (2).jpeg';
import Footer from './Footer';
import CartIcon from './CartIcon';

export default function Home1() {
  return (
    <div style={{ backgroundImage: `url(${background})` }}>
      {/* <div className="container my-5">
        <img src={makeup} alt="" />
      </div> */}
      <div className="container-fluid ">
        <div className="row">
            <div className="col-sm">
                <div className="row-sm">
                <div className="container maincar">
                  <div id="demo" class="carousel slide" data-ride="carousel" data-interval="2000">

                    <ul class="carousel-indicators">
                      <li data-target="#demo" data-slide-to="0" class="active"></li>
                      <li data-target="#demo" data-slide-to="1"></li>
                      <li data-target="#demo" data-slide-to="2"></li>
                    </ul>

                    <div class="carousel-inner">
                      <div class="carousel-item active">
                        <img src="https://images.indianexpress.com/2020/08/augmented-reality-for-lipstick-testing.jpg" className="mainimg img-fluid"></img>
                        </div>
                        <div class="carousel-item">
                        <img src="https://miro.medium.com/max/1400/1*NXvzo9qKeZYLdSb4r2tuig.jpeg" className="mainimg img-fluid"></img>
                        </div>
                        <div class="carousel-item">
                        <img src="https://d3ss46vukfdtpo.cloudfront.net/static/media/makeupar-6.6e0b53a8.jpg" className="mainimg img-fluid"></img>
                      </div>

                      <a class="carousel-control-prev" href="#demo" data-slide="prev">
                        <span class="carousel-control-prev-icon"></span>
                      </a>
                      <a class="carousel-control-next" href="#demo" data-slide="next">
                        <span class="carousel-control-next-icon"></span>
                      </a>
                    </div>
                    </div>
                  </div>
                    <h2 className="carousel-caption1">Augmented Reality Platform</h2>
                    <p className="carousel-caption1">With mirrAR you can virtually try on fashion and beauty products and experience how it feels to own them before the actual purchase.</p>
                    <br/>
                    <h2 className="carousel-caption1">Our Products</h2>
                </div>
            </div>
            
        </div>
    
    </div>
      <div className="container my-3">
      <div class="row">
        <div class="col d-flex justify-content-center mb-4">
        <Card/>
        </div>
        <div class="col d-flex justify-content-center mb-4">
        <Card1/>
        </div>
        <div class="col d-flex justify-content-center mb-4">
        <Card2/>
        </div>
      </div>        
      </div>
      {/* <Footer/> */}
      <CartIcon/>
      </div>
  );
}
