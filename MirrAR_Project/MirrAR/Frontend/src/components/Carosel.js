import React from 'react';

function Carosel() {
  return (
      <>
      <div id="carouselExampleControls" className="carousel slide" data-bs-ride="carousel" >
        <div className="carousel-inner">
            <div className="carousel-item active" style={{height: "550px"}}>
            <img src="https://d3ss46vukfdtpo.cloudfront.net/static/media/makeupar-6.6e0b53a8.jpg" style={{height: "550px"}} className="d-block w-100" alt="..."/>
            </div>
            <div className="carousel-item" style={{height: "550px"}}>
            <img src="https://miro.medium.com/max/1400/1*NXvzo9qKeZYLdSb4r2tuig.jpeg" style={{height: "550px"}} className="d-block w-100" alt="..."/>
            </div>
            <div className="carousel-item" style={{height: "550px"}}>
            <img src="https://images.indianexpress.com/2020/08/augmented-reality-for-lipstick-testing.jpg" style={{height: "550px"}} className="d-block w-100" alt="..."/>
            </div>
        </div>
        <button className="carousel-control-prev" type="button" data-bs-target="#carouselExampleControls" data-bs-slide="prev">
            <span className="carousel-control-prev-icon" aria-hidden="true"></span>
            <span className="visually-hidden">Previous</span>
        </button>
        <button className="carousel-control-next" type="button" data-bs-target="#carouselExampleControls" data-bs-slide="next">
            <span className="carousel-control-next-icon" aria-hidden="true"></span>
            <span className="visually-hidden">Next</span>
        </button>
    </div>
      </>
  );
}

export default Carosel;
