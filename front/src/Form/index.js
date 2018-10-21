import React        from 'react';
import { Redirect } from 'react-router-dom';
import Button       from '../Button';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    void(0),
        loading: false,
        showButton: false,
        imageType: 1
    }

    setRedirect = redirect => this.setState({redirect})

    handleSubmit = (event) => {

        var formData = new FormData();

        for (let i = 0; i < this.state.files.length; i++) {
            let file = this.state.files[i];

            formData.append('file', file);
        }

        this.setState({loading: true});

        fetch(
            'http://127.0.0.1:5000/v1/predict',
            {
                method: 'POST',
                body: formData
            }
        )
        .then(
            res => {
                try {
                   return res.json();
                } catch (err) {
                   throw `FETCH failed: ${res.status} ${res.statusText} ${err}`;
                }
            }
        )
        .then( res => this.setState({res, loading: false}) );

        event.preventDefault();
    }

    handleMouseMove = event => {
        var image;

        event.target.onmouseup = (e) => { e.target.onmousemove = null; return; };
        document.onmouseup     = (e) => { if (image) {image.onmousemove = null;} };

        event.target.onmousemove = (e) => {
            image = e.target;

            e.target.style.webkitFilter = `brightness(${
                50 + (e.clientX - e.target.offsetLeft)/(e.target.offsetWidth / 100)
            }%) contrast(${
                50 + (e.clientY - e.target.offsetTop)/(e.target.offsetHeight / 100)
            }%)`;

            !this.state.showButton && this.setState({ showButton: true })
        }
    }

    renderForm = (label, files) =>
        <form className="form__form" onSubmit={ this.handleSubmit }>
            <input
                type      = "file"
                name      = "data"
                id        = "data"
                className = "form__inputfile"
                onChange  = {
                    e => this.setState({
                            files: e.target.files,
                            label: e.target.value.split('\\').pop()
                        })
                }
            />
            <label htmlFor = "data">
                {
                    label ||
                    <span className="button">Select .zip file...</span>
                }
            </label>

            {
                files &&
                <input
                    type      = "submit"
                    value     = "Submit"
                    className = "button"
                />
            }
        </form>

    renderResults = (res, currentImage, showButton, imageType) =>
        <div className="form__results">
            <div className="form__results-list">
                {
                    res.map(
                        ({image, mask, processed, percent}, index) =>
                            <div
                                key       = {`img-${index}`}
                                tabIndex  = {index + 1}
                                className = "form__result"
                                onClick   = {() => this.setState({ currentImage: {image, mask, processed, percent, index} })}
                                onFocus   = {(e) => {
                                    document.querySelector('.form__image img').style.webkitFilter = '';
                                    this.setState({ currentImage: {image, mask, processed, percent, index}, showButton: false });
                                }}
                            >
                                <div>{`Image ${index}`}</div>
                                <div className = "form__result-certainty">
                                    {
                                        (percent > 0.75) &&
                                        `Certainty ${Math.round(percent  * 100)}%`
                                    }
                                </div>
                            </div>
                    )
                }
            </div>
            <div className="form__image">
                <div className="form__image-header">
                    <div className="form__image-meta">
                        <div className="form__image-label">{`Image ${currentImage.index || 0}`}</div>
                        <div>{`Certainty: ${(currentImage.percent * 100).toFixed(2)}%`}</div>
                    </div>
                    <div className="form__image-controls">
                        <Button
                            text="Original image"
                            mod={imageType === 1 ? 'active' : ''}
                            onClick={() => this.setState({imageType: 1})}
                        />
                        <Button
                            text="Heatmap"
                            mod={imageType === 2 ? 'active' : ''}
                            onClick={() => this.setState({imageType: 2})}
                        />
                        <Button
                            text="Nodules"
                            mod={imageType === 3 ? 'active' : ''}
                            onClick={() => this.setState({imageType: 3})}
                        />
                    </div>
                </div>
                <img
                    src       = {
                        `data:image/png;base64,${
                            imageType === 1
                                ? currentImage.image
                                : imageType === 2
                                    ? currentImage.mask
                                    : currentImage.processed
                        }`
                    }
                    draggable = {false}
                    onMouseDown = {(e) => this.handleMouseMove(e)}
                />
                {
                    showButton &&
                    <Button
                        text="Reset filter"
                        mod="reset"
                        onClick={() => {
                            document.querySelector('.form__image img').style.webkitFilter = '';
                            this.setState({ showButton: false });
                        }}
                    />
                }
            </div>
        </div>

    renderPredict = ({ redirect, files, label, res, loading, currentImage, showButton, imageType }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <section className="form">
                    <h1 className="form__header">Prediction</h1>
                    {
                        res
                            ? this.renderResults(res, currentImage || res[0], showButton, imageType)
                            : loading
                                ? <div className="form__loading">
                                    <div className="form__loading-text">loading</div>
                                    <svg width="100" height="30">
                                        <circle id="cLeft"   cx="20" cy="15" r="10" />
                                        <circle id="cCentre" cx="50" cy="15" r="10" />
                                        <circle id="cRight"  cx="80" cy="15" r="10" />
                                    </svg>
                                </div>
                                : this.renderForm(label, files)
                    }


                    { console.log(this.state) }
                </section>
    }

    render = () => this.renderPredict( this.state )

}