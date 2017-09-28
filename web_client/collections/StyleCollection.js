import Backbone from 'backbone';
import { LocalStorage } from 'backbone.localstorage';

import StyleModel from '../models/StyleModel';

const StyleCollection = Backbone.Collection.extend({
    model: StyleModel,
    localStorage: new LocalStorage('histomicstk.draw.style')
});

export default StyleCollection;
