import { useEffect, useState } from 'react';
import './App.css';
import LoadingCircle from './components/LoadingCircle';
import Home from './components/Home';
import Lips from './components/Lips';
import Shirt from './components/Shirt';
import Tryout from './components/Tryout_Lips';
import Tryout_shirt from './components/Tryout_shirt';
import { BrowserRouter as Router, Switch, 
  Route, Redirect,} from "react-router-dom";
import Tryout_necklace from './components/Tryout_necklace';
import Necklace from './components/Necklace';


function sleep(milliseconds) {
  const date = Date.now();
  let currentDate = null;
  do {
    currentDate = Date.now();
  } while (currentDate - date < milliseconds);
}

function returnTrue() {
  sleep(7000)
  return true;
}

function App() {
  const[post, setPost] = useState(null)


  useEffect(() =>{
    fetch("https://jsonplaceholder.typicode.com/posts/1").then(res => res.json()
      .then(res =>{
        sleep(7000)
        const {body} = res
        setPost(body)
      })
    )
  })


  return (
    <div className="App"> 
      <Router>
        <Switch>
          <Route exact path="/" component={post ? Home: LoadingCircle}/>
          <Route path="/lipproducts" component={Lips} />
          <Route path="/tshirt" component={Shirt} />
          <Route path="/necklace" component={Necklace} />
          <Route path="/tryout" component={Tryout} />
          <Route path="/tryout_shirt" component={Tryout_shirt} />
          <Route path="/tryout_necklace" component={Tryout_necklace} />
          <Redirect to="/" />
        </Switch>
      </Router>
      {/* <Navbar/> */}
      {/* {post ? <Categories />: <LoadingCircle/>} */}
      {/*<Tryout/>*/}
      {/* <LipProducts/> */}
    </div>
  );
}

export default App;
